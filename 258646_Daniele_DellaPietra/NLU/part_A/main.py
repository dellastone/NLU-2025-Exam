"""
## Part 2.A
As for the first part of the project (LM), you have to apply these two modifications incrementally. Also in this case you may have to play with the hyperparameters and optimizers to improve the performance. 

Modify the baseline architecture Model IAS by:
- Adding bidirectionality
- Adding dropout layer

**Intent classification**: accuracy <br>
**Slot filling**: F1 score with conll

***Dataset to use: ATIS***

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
            name = exp['name'],
            train=exp['train'],
            test=False,
            use_dropout=exp['use_dropout'],
            bidirectional=exp['bidirectional'],
            dropout=exp['dropout'],
            learning_rate=exp['learning_rate'],
            batch_size=exp['batch_size'], 
            hidden_dim=exp['hidden_dim'],
            embedding_dim=exp['embedding_dim'],
            clip=exp['clip'],
            patience=exp['patience'],
            epochs=exp['epochs'],
            save_model=args.save_model,
            wandb=args.wandb
        )

        start_training(exp_args)