# MARL Comparative Study Framework (Simple World Comm)

This scaffold is designed for a comparative study of:
- IPPO
- MAPPO
- DQN / Double DQN
- DDPG
- PSO

Environment:
- `mpe2.simple_world_comm_v3`

The framework shares:
- environment creation
- evaluation protocol
- logging
- metrics
- config management

Each algorithm has its own file under `algorithms/`.

## Suggested workflow
1. Finish `envs.py` and make sure the environment runs.
2. Implement IPPO and MAPPO first.
3. Add DQN / Double DQN.
4. Add DDPG.
5. Add PSO.
6. Use the same logging/evaluation protocol for all methods.

## Run example
```bash
python main.py --algo ippo
python main.py --algo mappo
python main.py --algo dqn
python main.py --algo ddpg
python main.py --algo pso
```
