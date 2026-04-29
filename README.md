# marl_comm_v2 — 项目交接 README

## 一句话目标

在 `mpe2.simple_world_comm_v3` 上证明 **E1（开通信）显著优于 E2（关通信）**，作为论文核心论点的实证支持。

---

## 1. 项目背景

### 1.1 来自上一个项目（marl_comm_framework）的发现

之前的 Study A 实验得到了一个 null result：4 种 ablation（full / no_comm / no_leader_pos / blind）的 final reward 都在 ~231-237 之间，统计上无差异。但 ABCD 分析显示 leader 学到了一个 74% 可预测的 state→message encoding。结论是 "encoder 学到了，decoder 没学到"。

**那个项目我们已经完成了。这个新项目 marl_comm_v2 是为了让"通信确实有用"被实验证明。**

### 1.2 marl_comm_v2 的设计思路

不再依赖 emergent communication 自己学出来——直接给 message **显式语义**，并给 reward 设计**明确的合作激励**：

- **Message 语义（ABCD）**：
  - A = focus good_0
  - B = focus good_1
  - C = rush forest_0
  - D = rush forest_1
- **架构改动**：
  - Leader 看到自己 obs + last_msg one-hot + forest_flags → 42 维
  - Adversary 看到自己 obs + message one-hot → 38 维
  - Adversary actor 是 `AdversaryAlphaActor`：trunk → (own_logits[5], alpha[1])，最终 logits = α × own_logits + (1-α) × plan_bias
  - **plan_bias** 是基于规则的 5 维方向向量：根据 message 指示的目标，给出朝它走的方向 logit
- **Reward shaping**：合作围捕得分远高于单 adv 行动

---

## 2. 三个核心实验条件

| Condition | message | plan_bias | alpha | 用意 |
|---|---|---|---|---|
| `E1_full` | 开 | 开 | 学习 (init=0.5) | 通信 + 规则提示 + 自适应权重 |
| `E2_no_comm` | 关 | 关 | 不用 | 没有任何通信信号 |
| `E3_fixed_alpha` | 开 | 开 | 固定 0.5 | 测试 α 学习的贡献 |

**核心假设**：E1 应该显著优于 E2。如果不能，论文的 thesis 站不住。

---

## 3. 文件结构

```
marl_comm_v2/
├── config.py              # 所有超参（reward weights、网络架构、训练参数）
├── envs.py                # WorldCommEnv 包装器，含 reward shaping、obs augment、ABCD 规则
├── main.py                # CLI 入口，--condition E1_full/E2_no_comm/E3_fixed_alpha
├── experiment.py          # 训练 loop，含 EVAL 逻辑（含 catches/encircle 指标）
├── sanity_compare.py      # 批量跑 E1+E2 各 500 ep 用于快速 sanity check
├── run_experiments.py     # 完整批量运行（E1/E2/E3 × 多 seeds × 4000 ep）
├── visualize.py           # 看每步 reward 分解 + 生成 GIF
├── frozen_policy.py       # 加载冻结的 standard good 模型
├── test_action_decode.py  # 验证 leader action 解码（a // 5 = say_idx）
└── algorithms/
    └── mappo.py           # MAPPO 实现，含 AdversaryAlphaActor
```

### 3.1 各文件做什么

**config.py** — 所有超参。最重要的是 reward weights、`max_cycles`、`plan_bias_strength`、`alpha_init`。

**envs.py** — 环境包装器，干了几件事：
1. 包装 mpe2.simple_world_comm_v3
2. 给 leader obs 加 last_msg + forest_flags（28 → 42）
3. 给 adv obs 加 message one-hot（34 → 38）
4. 实现 `_compute_correct_message`：根据当前 good 位置和 forest 状态，规则化决定"正确"message
5. 实现 `compute_plan_bias`：根据 message 给 adv 一个 5 维方向向量
6. 实现 `_compute_structured_rewards`：所有 reward 项的核心计算
7. 跟踪 `_goods_caught_ever`（哪些 good 被首次抓到）
8. `set_eval_mode(True)` 标志：EVAL 时跳过教学奖励（详见 §4.3）

**experiment.py** — 训练 loop。每个 episode 跟踪：
- `adv_return` / `good_return`（平均奖励）
- `adv_total`（4 adv 总和）
- `catches`（这个 episode 抓到几个 good）
- `encircle_steps`（≥3 个 adv 围着某个 good 的步数）

EVAL 输出格式：
```
[EVAL 100] team_total= 218.0  adv_avg= 54.51±14.91  catches=1.60/2  encircle_steps=0.2/50  good=-26.15
```

**algorithms/mappo.py** — MAPPO + Role parameter sharing。三组网络：
- Leader（独立）
- Normal adversary（3 个共享一套 actor + critic）
- Good（独立）

`AdversaryAlphaActor` 是关键：它有一个 alpha_head 输出 [0,1] 标量，用 `final_logits = α × own_logits + (1-α) × plan_bias` 混合。

**visualize.py** — `--breakdown` 显示每步每个 reward 分量。`--gif` 生成回放。

**sanity_compare.py** — 调试工具，500 ep 快速跑 E1+E2 看趋势。**这是当前主要用来验证设计的工具**。

---

## 4. 当前 Reward 设计（最重要章节）

每个 normal adv 每步收到的 reward：

```
R_total = R_distance + R_capture + R_encircle + R_progress + R_pin_assist
        + R_team_bonus + R_role_align (训练时)
```

Leader 额外收到：

```
R_leader = R_thrash (训练时) + R_correct_message (训练时)
```

### 4.1 实战奖励（训练 + EVAL 都算）

| 项 | 数值 | 触发条件 |
|---|---|---|
| `R_distance_coef` | -0.05 | 每步乘以 adv 到最近**未被抓的** good 的距离 |
| `R_capture` | +10 | adv 撞到未被抓的 good（一次性） |
| `R_encircle_per_extra` | +4 | n 个 adv (n>=3) 围在 0.4 半径内 → 每人 +(n-1)*4 |
| `R_progress_first` | +30 | 给**第一个被抓 good 的 killer**（撞到的那个 adv） |
| `R_progress_second` | +40 | 给**第二个被抓 good 的 killer** |
| `R_progress_team_share` | +15 | 给围捕圈内（除 killer 外）的队友（split incentive） |
| `R_pin_assist` | +10 | 给围捕圈内（除 killer 外）的队友 |
| `R_team_bonus` | +5 | 给所有 adv（任何首抓发生时） |

### 4.2 教学奖励（仅训练时；EVAL 跳过）

| 项 | 数值 | 触发条件 |
|---|---|---|
| `R_role_align` | +1 | adv 当前 action == argmax(plan_bias) |
| `R_message_thrash_penalty` | -3 | leader 切换 message 但 forest 状态/catch 状态没变（防乱发） |
| `R_correct_message` | +5 | leader 切换到"正确" message（switch-only，详见 envs.py 的 meaningful-switch 逻辑） |

### 4.3 为什么训练 vs EVAL 要分开

之前发现：E1 拿 R_role_align 每步 +1（共 +50/episode），E2 没 plan_bias 拿不到这个分。这让 E1 的 EVAL 数值虚高，看上去有优势但实际抓的 good 比 E2 少。

现在的设计：
- **训练时所有 reward 都算**——给 PPO 学习信号
- **EVAL 时只算实战奖励**——E1 vs E2 公平对比

EVAL 的关键指标是 **catches/2**（事件计数，永远公平），其次是 EVAL 时的 team_total。

### 4.4 Reward 量级核算（50 步理想合作）

| 项 | killer 累积 | 围捕圈内 adv 累积 |
|---|---|---|
| Distance shaping | -2.5 | -2.5 |
| Encircle 10 步 (n=4) | +120 | +120 |
| Capture 1st good | +10 | 0 |
| Progress 1st | +30 | +15 |
| Pin_assist 1st | 0 | +10 |
| Team_bonus 1st | +5 | +5 |
| Capture 2nd good | +10 | 0 |
| Progress 2nd | +40 | +15 |
| Pin_assist 2nd | 0 | +10 |
| Team_bonus 2nd | +5 | +5 |
| Role_align 50 步（仅训练）| +50 | +50 |
| **训练时总计** | **~277** | **~227** |
| **EVAL 时总计** | **~227** | **~177** |

---

## 5. 当前问题（需要解决的）

### 5.1 现象

跑 sanity_compare（500 ep × 2 conditions）发现 **E1 没有显著优于 E2**：

| 指标 | E1 EP500 | E2 EP500 | 期望 |
|---|---|---|---|
| catches/2 | 1.10 | 1.40 | E1 > E2 |
| encircle_steps/50 | 0.3 | 1.0 | E1 > E2 |
| team_total | 174 | 110 | E1 > E2 |

E1 的 team_total 高仅仅是因为 R_role_align 的 +50/episode（已被 EVAL 修正过滤），**catches 反而 E2 更多**。

### 5.2 我们的诊断（可能是对的，可能不是）

#### 假设 1：plan_bias 让所有 4 个 adv 同向移动，破坏自然包围

包围本来需要 4 个 adv 从不同方向夹击。但 plan_bias 给所有 adv **同一个方向**——比如 message=A，所有 adv 都收到"朝 good_0 走"。结果 4 个 adv 像追逐车队一样跟着 good_0，从同一方向追，围不上。

E2 没 plan_bias，每个 adv 看自己 obs，自然从不同位置逼近，反而能围。

#### 假设 2：任务对称性

good_0 和 good_1 对称、forest_0 和 forest_1 对称。leader 学到的策略容易**坍缩**到只用 2 个 message（比如只用 A 和 C，从不用 B 和 D）。这样 message 信息量不足，对 adv 帮助有限。

#### 假设 3：500 ep 太短

之前 marl_comm_framework 跑 4000 ep 才看到收敛。500 ep 可能还没学到 message 的用法。但反复实验显示 EP100 就接近峰值，之后波动，500 ep 应该已经够看趋势。

### 5.3 我们尝试过的（都没成功）

| 尝试 | 结果 |
|---|---|
| 把 R_progress 改成只给 killer | reward 数值减半，训练弱化 |
| max_cycles 50 → 25（时间压力） | 25 步走不到 good，更差 |
| plan_bias_strength 4 → 2 → 1 → 0 | strength=0 时 actor 被 α 缩放，几近随机；其他值无显著改善 |
| R_role_align 1 → 0.5 → 0 | 都没救 |
| R_correct_message 5 → 0 | leader 不发 message 多样性，adv 没信号学 |
| 加 R_pin_assist + R_team_bonus | 数值起来了但 catches 没起来 |
| 加 eval_mode 区分训练/EVAL reward | 让对比公平了，但 E1 仍未赢 E2 |

**最近一次配置**（README 描述的）回到 50 步 + plan_bias=4 + R_progress 加 team_share，是当前 README 里写的设定，**还没充分测试过**。

---

## 6. 给你的几个改造方向

按推荐度排序：

### 方向 A：让 plan_bias 对不同 adv 给不同方向（最有希望）

**问题**：当前 plan_bias 给 4 个 adv 同一个方向。

**思路**：让 plan_bias 也带 **agent identity 信息**——比如 adv_0 收到的 plan_bias 是"从右上方逼近 good_0"，adv_1 收到的是"从左下方逼近 good_0"。

具体可以这样：
- 在 `compute_plan_bias` 里，根据 adv 当前位置相对 target 的方位，给出一个**"包围"方向**而不是"直奔"方向
- 例如 adv 在 target 的东北方 → plan_bias 向南偏（让它绕到东南）

这相当于把"要去 good_0"和"该从哪个方向去"分开。Message 仍然只传 ABCD，adv 自己根据相对位置决定具体走法。

文件改动：`envs.py` 里 `compute_plan_bias` 函数（约第 350 行附近）。

### 方向 B：放弃 plan_bias，让 message 只通过 obs 影响 adv

**问题**：plan_bias 是规则化的，太强会绑架 adv 行为。

**思路**：
- `plan_bias_strength = 0`，但同时**修改 actor 架构**——不要 `final_logits = α * own_logits + (1-α) * plan_bias` 这种结构。直接 `final_logits = own_logits`（普通 actor）。
- adv 仍然能看到 obs[34:38] 的 message one-hot，让网络自己学怎么用。

这是真正的 emergent communication（带显式 ABCD 监督）。但要让它 work，需要：
1. R_correct_message 保留（训练 leader 发不同 message）
2. 训练时间可能要 2000+ ep 才看到 message 被用

文件改动：`algorithms/mappo.py` 的 `AdversaryAlphaActor` 类，可以加一个 flag 让它退化成普通 actor。

### 方向 C：减小任务对称性

**问题**：good_0 = good_1，forest_0 = forest_1，leader 容易学到只用 A+C 不用 B+D。

**思路**：
- 让 good_1 比 good_0 多一些价值（比如 R_progress_second_good_bonus = +20）
- 或者让 forest_0 和 forest_1 大小不同（环境层修改，可能违反 prof 的 num_forests=2 要求）

最简单：在 reward 里加 "leader 使用 message 多样性的 entropy bonus"——leader 一个 episode 里 4 种 message 的分布越接近均匀，bonus 越高。

文件改动：`envs.py` 在 reward 计算的 leader 部分加一项。

### 方向 D：换思路，承认 null result，从 Study A 角度写论文

如果发现这个问题真的不是 reward 能解决的（可能是任务本身就不需要通信），那就承认：

> "在我们的设置下，cooperative reward shaping 已经足够强让 adv 团队学会包围，单 adv 自身的 obs 已经足够引导行动；显式 communication 在这种条件下并未提供可测量的额外信息增益。这与 Mordatch & Abbeel (2017) 在更难任务上观察到的 emergent communication 形成对比。"

这是诚实的科研表述，比硬调出"E1 > E2"有价值。

---

## 7. 怎么开始

### 7.1 环境

```bash
conda create -n stats402 python=3.10
conda activate stats402
pip install torch numpy pandas matplotlib scikit-learn pettingzoo mpe2 gymnasium pygame
```

### 7.2 跑 sanity check（10 分钟出结果）

```cmd
python sanity_compare.py
```

输出每个 condition 的 5 个 EVAL 行。重点看最后一行的 **catches** 和 **encircle_steps**。

### 7.3 跑单个 condition（单独调试用）

```cmd
python main.py --condition E1_full --seed 42 --run_name debug --total_episodes 500
```

### 7.4 看每步 reward 分解（调试用）

```cmd
python visualize.py --checkpoint outputs/models/sanity_mappo_E1_full_s42/final --condition E1_full --breakdown --n_episodes 5
```

注意：跑 E2 ckpt 时一定要传 `--condition E2_no_comm`，否则会因为 alpha_head 不存在而报错（已加 `strict=False` 兼容，但 condition 还是要对）。

### 7.5 跑生成 GIF 看 adv 行为

```cmd
python visualize.py --checkpoint outputs/models/sanity_mappo_E1_full_s42/final --condition E1_full --gif --n_episodes 3
```

### 7.6 正式实验（如果 sanity 过了）

```cmd
python run_experiments.py
```

跑 E1/E2/E3 × 3 seeds × 4000 ep，约 2.5 小时。

---

## 8. 成功标准

E1 EP500 catches 比 E2 高 >= 0.3（比如 1.5/2 vs 1.0/2），且 team_total 高 30%+ 即可视为 sanity 通过。然后跑正式 4000 ep × 3 seeds，三个 seed 平均一致即论文核心结论成立。

如果 4000 ep 跑出来 E1 仍 <= E2，那就走方向 D，老实写 null result。

---

## 9. 几个我们已经踩过的坑（避免重蹈覆辙）

1. **Leader action 解码**：`a in [0, 19]`，`say = a // 5`，`move = a % 5`。早期写成 `a % 4` 是 bug。`test_action_decode.py` 验证过。

2. **plan_bias_strength = 0 是灾难**：actor 输出会被 α 缩放到 (0,1)*own_logits，softmax 几近平均，adv 接近随机。如果想关 plan_bias，必须同时改 actor 结构（方向 B）。

3. **E2 加载 visualize 报错**：E2 训练时没保存 alpha_head，加载时 strict=True 会报 missing key。已改为 strict=False。

4. **good_0 / good_1 的 obs 顺序**：mpe2 的 obs 是按 agent 字典序排的。检查 `_reconstruct_adv_base_obs` 时要小心。

5. **catches 计数**：用 `_goods_caught_ever` set，避免一个 good 被反复计数。set 在 reset 时清空。

6. **eval_episodes 数量**：默认 10。EVAL 期间 4 个 adv 全是 deterministic（argmax），所以方差较低，但仍有 environmental 随机性（reset 时 good 位置随机）。

---

## 10. 联系

如果跑出来 E1 显著超过 E2，跑完整 4000 ep 实验（约 2.5 小时），把 outputs/logs 文件夹的 CSV 发回来。

如果尝试了方向 A/B/C 都不 work，跑完一组对比测试，把 sanity_compare 输出发回来，我们一起决定走方向 D。
