# NLU Assignment 1 — Part B

This repository contains the code and configs for **Assignment 1 (Part B)**.  
Experiments are defined in a YAML file and executed via a single entrypoint.

---

## Quick start

Run all experiments listed in `experiments_config.yaml`:

```bash
python main.py
```

**Optional flags**
- `--save_model` — save the best checkpoint of each run to `bin/{name}_model.pth`
- `--wandb` — log metrics to Weights & Biases
