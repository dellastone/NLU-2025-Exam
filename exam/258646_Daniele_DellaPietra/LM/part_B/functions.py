# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
import argparse
from utils import PennTreeBank, Lang, read_file, collate_fn
from functools import partial
from torch.utils.data import DataLoader
import torch
from model import LM_model
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import math
import numpy as np
import wandb
def load_data(device='cuda', batch_size=64):
    """
    Load the dataset
    """
    print('Loading the dataset')
    
    
    train_raw = read_file("../dataset/PennTreeBank/ptb.train.txt")
    dev_raw = read_file("../dataset/PennTreeBank/ptb.valid.txt")
    test_raw = read_file("../dataset/PennTreeBank/ptb.test.txt")
    
    lang = Lang(train_raw, ["<pad>", "<eos>"])
    
    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], device = device),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=2*batch_size, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"], device = device))
    test_loader = DataLoader(test_dataset, batch_size=2*batch_size, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"],device = device))
    
    return train_loader, dev_loader, test_loader, lang

def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)
                    
                    
def train_loop(data, optimizer, criterion, model, clip=5):
    model.train()
    loss_array = []
    number_of_tokens = []
    
    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        output = model(sample['source'])
        loss = criterion(output, sample['target'])
        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step() # Update the weights
        
    return sum(loss_array)/sum(number_of_tokens)

def eval_loop(data, eval_criterion, model):
    model.eval()
    loss_to_return = []
    loss_array = []
    number_of_tokens = []
    # softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            output = model(sample['source'])
            loss = eval_criterion(output, sample['target'])
            loss_array.append(loss.item())
            number_of_tokens.append(sample["number_tokens"])
            
    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    return ppl, loss_to_return

def start_training(args):
    """
    Start training the model
    """
    print(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device {device}')

    #initilize wandb
    wandb.init(entity='della_stone',project="NLU", config=args)
    
    #load dataset
    batch_size = args.batch_size
    train_dataloader, dev_dataloader, test_dataloader, lang = load_data(device, batch_size)
    vocab_len = len(lang.word2id)
    #save arguments
    model_name = args.model
    
    hidden_dim = args.hidden_dim
    emb_dim = args.embedding_dim
    lr = args.learning_rate
    clip = args.clip
    patience = args.patience
    epochs = args.epochs
    use_dropout = args.use_dropout
    optimizer = args.optimizer
    dropout = args.dropout
    #load model
    
    if model_name == 'RNN':
        print('Training RNN model')
    elif model_name == 'LSTM':
        print('Training LSTM model')
    else:
        print('Model not implemented')
        exit()
    model = LM_model(model_name, hidden_dim, emb_dim, vocab_len, use_dropout, pad_index=lang.word2id["<pad>"], emb_dropout=dropout, out_dropout=dropout).to(device)
    #print the model number of parameters
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    model.apply(init_weights)

    if optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')   
    
    print('Training the model')
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None
    pbar = tqdm(range(1,epochs))
    #If the PPL is too high try to change the learning rate
    for epoch in pbar:
        loss = train_loop(train_dataloader, optimizer, criterion_train, model, clip)    
        # Sample and log metrics every epoch (or adjust frequency if needed)
        sampled_epochs.append(epoch)
        train_loss = np.asarray(loss).mean()
        losses_train.append(train_loss)
        
        ppl_dev, loss_dev = eval_loop(dev_dataloader, criterion_eval, model)
        dev_loss = np.asarray(loss_dev).mean()
        losses_dev.append(dev_loss)
        
        pbar.set_description("PPL: %f" % ppl_dev)
        
        # Log metrics to WandB
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "dev_loss": dev_loss,
            "dev_ppl": ppl_dev
        })
        
        # Early stopping and best model saving based on dev perplexity
        if ppl_dev < best_ppl:  # the lower, the better
            best_ppl = ppl_dev
            best_model = copy.deepcopy(model).to(device)
            patience = args.patience
        else:
            patience -= 1

        if patience <= 0:  # Early stopping with patience
            break

    best_model.to(device)
    final_ppl,  _ = eval_loop(test_dataloader, criterion_eval, best_model)    
    print('Test ppl: ', final_ppl)
    if args.save_model:
        print('saving model')
        #create the date_time folder inside model_bin
        import os
        from datetime import datetime
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        os.makedirs(f"model_bin/{dt_string}")
        torch.save(best_model.state_dict(), f"model_bin/{dt_string}/{model_name}_model.pth")

    wandb.finish()


def get_arguments():
    """
    Get arguments from command line
    """
    
    parser = argparse.ArgumentParser(
        prog='main.py',
        description='Get args',
    )
    
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train the model',
        default=False
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test the model',
        default=False
    )
    
    parser.add_argument(
        '--model',
        type=str,
        help='Model to use. Choose between RNN, LSTM',
        default='RNN'
    )
    
    parser.add_argument(
        '--use_dropout',
        action='store_true',
        help='Use dropout',
        default=False
    )
    
    parser.add_argument(
        '--optimizer',
        type=str,
        help='Optimizer to use',
        default='SGD'
    )
    
    parser.add_argument(
        '--hidden_dim',
        type=int,
        help='Hidden dimension of the model',
        default=128
    )
    
    parser.add_argument(
        '--embedding_dim',
        type=int,
        help='Embedding dimension of the model',
        default=128
    )
    
    parser.add_argument(
        '--learning_rate',
        type=float,
        help='Learning rate of the model',
        default=1.0
    )
    
    parser.add_argument(
        '--clip',
        type=float,
        help='Gradient clipping',
        default=5
    )
    
    parser.add_argument(
        '--patience',
        type=int,
        help='Patience for early stopping',
        default=5
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of epochs',
        default=100
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        help='Batch size',
        default=256
    )

    parser.add_argument(
        '--save_model',
        action='store_true',
        help='Save the model',
        default=False
    )

    parser.add_argument(
        '--dropout',
        type=float,
        help='Dropout rate',
        default=0.1
    )
    
    return parser.parse_args()
    