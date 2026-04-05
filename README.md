# STATS402 MARL Scaffold

Minimal Stage-1 scaffold for comparing IPPO vs MAPPO on PettingZoo MPE Simple Spread.

## Install

```bash
pip install torch numpy pettingzoo pygame gymnasium supersuit
```

If your environment uses `mpe2`, keep that installed too.

## Run

```bash
python main.py --algo ippo
python main.py --algo mappo
```

## Notes

- This starter code assumes **discrete** action space.
- It is intentionally focused on **Simple Spread first**.
- `coverage_efficiency` is a temporary proxy metric right now; replace it later with a real environment-specific metric.
