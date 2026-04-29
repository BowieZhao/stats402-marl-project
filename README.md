# MARL Communication — Final Design

## 项目结构

```
new_project/
├── config.py              # 配置 (新)
├── envs.py                # 环境包装 + plan_bias + reward (新)
├── experiment.py          # Runner (新)
├── frozen_policy.py       # Frozen good 加载 (新, 兼容旧 ckpt)
├── main.py                # CLI 入口 (新)
├── run_experiments.py     # 批量运行 E1/E2/E3 (新)
├── algorithms/
│   ├── __init__.py        # 从原项目复制
│   ├── mappo.py           # MAPPO + AdversaryAlphaActor (新)
│   ├── dqn.py             # 从原项目复制（不在新实验里用）
│   ├── ddpg.py            # 从原项目复制
│   └── pso.py             # 从原项目复制
├── analyze_comm.py        # 从原项目复制（用于 E1 的 ABCD 分析）
├── plot_all.py            # 从原项目复制（绘图）
└── outputs/
    └── models/
        └── standard_good_mappo_full_coop_s42/   # 复用旧 standard good
```

## 从原项目复制的文件清单

将以下文件**直接复制**到新文件夹：
- `algorithms/__init__.py`
- `algorithms/dqn.py`（备用 baseline）
- `algorithms/ddpg.py`（备用 baseline）
- `algorithms/pso.py`（备用 baseline）
- `analyze_comm.py`
- `plot_all.py`
- `outputs/models/standard_good_mappo_full_coop_s42/`（整个目录）

## 设计要点

### Message ABCD 显式语义
- A: focus good_0
- B: focus good_1
- C: rush forest_0
- D: rush forest_1

### Plan_bias rule-based decoder
把 message + adv obs 翻译为 5 维方向 logit bias（朝目标移动）。

### Adversary actor: alpha-gated logits combo
```
final_logits = α × own_logits + (1 - α) × plan_bias
```
α ∈ [0, 1] 是 adv 自己学的标量信任度。

### Structured Reward
```
R = -0.05 × dist
  + 10  × my_collision
  + (n-1)*4 × in_encircle_group (≥3 advs at 0.4 radius)
  + 30  × first_team_catch_of_good
  + 1   × action_matches_plan_bias
  + (-3 × leader_unjustified_thrash)  # leader only
```

### Obs 维度
- Leader: 34 + 4 (last_msg) + 4 (forest_flags) = 42
- Adversary: 34 + 4 (message_one_hot) = 38
- Good: 28 (不变)

## 实验设计

| Condition  | Message    | Plan_bias | α          |
|------------|------------|-----------|------------|
| E1_full    | 学习       | ✓         | 学习       |
| E2_no_comm | 屏蔽       | ✗         | N/A        |
| E3_no_alpha| 学习       | ✓         | 0.5 固定   |

## 运行步骤

### 1. Sanity check
跑一个 E1 短训练验证代码 work：
```bash
python main.py --condition E1_full --seed 42 --run_name sanity \
    --total_episodes 500 \
    --frozen_good outputs/models/standard_good_mappo_full_coop_s42/final
```

预期：
- 不报错跑完 500 ep
- adv reward 应该有上升趋势
- actor_loss/critic_loss 不发散

### 2. 跑全部实验
```bash
python run_experiments.py --frozen_good outputs/models/standard_good_mappo_full_coop_s42/final
```

跑 9 次（3 conditions × 3 seeds），约 2.5 小时。

### 3. 分析
- 看 outputs/logs/final_mappo_E*_*_*.csv 里的 reward 曲线
- 对比 E1 / E2 / E3 final reward (mean ± std over 3 seeds)
- 对 E1 的 checkpoint 跑 analyze_comm.py 看 ABCD 是否还有结构

## 注意事项

1. **新 reward 量级和旧实验不可比**：新 reward 范围更大（围捕奖励），旧 standard good 是用旧 reward 训的，**理论上应该用新 reward 重训 standard good**。但如果先想快速看效果，可以先用旧 standard good 跑 sanity，看到正向信号再考虑重训。

2. **alpha 学习需要监控**：在训练日志里观察 alpha 的演化是有意义的（可以加额外 logging 在 mappo.py 里）。

3. **如果 E1 < E2**：说明 message + plan_bias 在拖累训练，可能是 plan_bias_strength 太大或 alpha_init 偏低导致 adv 早期被 plan_bias 主导。
