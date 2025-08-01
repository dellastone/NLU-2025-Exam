# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py
"""
## Part 2.B

Adapt the code to fine-tune a pre-trained BERT model using a multi-task learning setting on intent classification and slot filling. 
You can refer to this paper to have a better understanding of how to implement this: https://arxiv.org/abs/1902.10909. In this, one of the challenges of this is to handle the sub-tokenization issue.

*Note*: The fine-tuning process is to further train on a specific task/s a model that has been pre-trained on a different (potentially unrelated) task/s.


The models that you can experiment with are [*BERT-base* or *BERT-large*](https://huggingface.co/google-bert/bert-base-uncased). 

**Intent classification**: accuracy <br>
**Slot filling**: F1 score with conll

***Dataset to use: ATIS***

"""
# Import everything from functions.py file
import yaml
import argparse
from functions import *  # your existing imports

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='experiments_config.yaml',
                        help='YAML file listing all experiments')
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
            lr=exp['learning_rate'],
            batch_size=exp['batch_size'], 
            clip=exp['clip'],
            patience=exp['patience'],
            epochs=exp['epochs'],
            data_dir=exp['data_dir'],
            bert_model=exp['bert_model'],
            max_len=exp['max_len'],
        )

        # call your existing training launcher
        start_training(exp_args)
        
