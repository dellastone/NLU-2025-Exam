
"""
## Part 1.B
**Mandatory requirements**: For the following experiments the perplexity must be below 250 (***PPL < 250***) and it should be lower than the one achieved in Part 1.1 (i.e. base LSTM).

Starting from the `LM_RNN` in which you replaced the RNN with a LSTM model, apply the following regularisation techniques:
- Weight Tying 
- Variational Dropout (no DropConnect)
- Non-monotonically Triggered AvSGD 

These techniques are described in [this paper](https://openreview.net/pdf?id=SyyGPP0TZ).

"""
import yaml
import argparse
from functions import *  # your existing imports

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='experiments_config.yaml',
                        help='YAML file listing all experiments')
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Load all experiments
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    for exp in cfg['experiments']:
        print(f"\n--- Running experiment: {exp['name']} ---")
        # build a namespace with defaults + override from the yaml
        exp_args = argparse.Namespace(
            train=True,
            test=False,
            name=exp['name'],
            model=exp['model'],
            optimizer=exp['optimizer'],
            use_dropout=exp['use_dropout'],
            dropout=exp['dropout'],
            learning_rate=exp['learning_rate'],
            batch_size=exp['batch_size'], 
            hidden_dim=exp['hidden_dim'],
            embedding_dim=exp['embedding_dim'],
            nt_asgd=exp['nt_asgd'],
            clip=5,
            patience=5,
            epochs=500,
            save_model=args.save_model,
            wandb=args.wandb
        )

        # call your existing training launcher
        start_training(exp_args)
  