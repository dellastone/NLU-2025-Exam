import os
import argparse
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from transformers import BertTokenizerFast, DataCollatorForTokenClassification
from transformers import get_linear_schedule_with_warmup
from utils import load_data, Lang, device, ATISBERTDataset
from model import BertForATISMultiTask
from conll import evaluate
from sklearn.metrics import accuracy_score
import torch.optim as optim
from collections import Counter
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings

# Suppress HF future warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def load_dataloader(data_dir, batch_size=32, max_len=128, model_name='bert-base-uncased'):
    """
    Loads ATIS train/test, creates a stratified dev split from train, and returns BERT dataloaders + Lang + tokenizer
    """
    # 1) Load raw splits
    train_raw = load_data(os.path.join(data_dir, 'train.json'))
    test_raw  = load_data(os.path.join(data_dir, 'test.json'))

    # 2) Create train/dev split stratified on intents
    intents = [x['intent'] for x in train_raw]
    count_y = Counter(intents)
    inputs, labels, mini_train = [], [], []
    for idx, intent in enumerate(intents):
        if count_y[intent] > 1:
            inputs.append(train_raw[idx]); labels.append(intent)
        else:
            mini_train.append(train_raw[idx])
    X_train, X_dev, _, _ = train_test_split(
        inputs, labels, test_size=0.1, random_state=42,
        stratify=labels, shuffle=True
    )
    X_train.extend(mini_train)
    train_raw, dev_raw = X_train, X_dev

    # 3) Build Lang mappings
    intents_set = set([x['intent'] for x in train_raw + dev_raw + test_raw])
    slots_set   = set(sum([ex['slots'].split() for ex in train_raw + dev_raw + test_raw], []))
    lang = Lang([], intents_set, slots_set, cutoff=0)

    # 4) Tokenizer and dataset
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    train_ds = ATISBERTDataset(train_raw, tokenizer, lang.slot2id, lang.intent2id, max_length=max_len)
    dev_ds   = ATISBERTDataset(dev_raw,   tokenizer, lang.slot2id, lang.intent2id, max_length=max_len)
    test_ds  = ATISBERTDataset(test_raw,  tokenizer, lang.slot2id, lang.intent2id, max_length=max_len)

    # 5) Collator to pad batches dynamically
    collator = DataCollatorForTokenClassification(tokenizer, label_pad_token_id=-100)

    # 6) DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collator)
    dev_loader   = DataLoader(dev_ds,   batch_size=batch_size, shuffle=False, collate_fn=collator)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, collate_fn=collator)

    return train_loader, dev_loader, test_loader, lang, tokenizer


def evaluate_bert_epoch(dataloader, model, tokenizer, lang):
    model.eval()
    all_refs, all_hyps = [], []
    intent_preds, intent_gts = [], []
    with torch.no_grad():
        for batch in dataloader:
            labels = batch.pop('slot_labels')
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            slot_ids = outputs['slot_logits'].argmax(-1).cpu()
            intent_ids = outputs['intent_logits'].argmax(-1).cpu().tolist()
            intent_preds.extend(intent_ids)
            intent_gts.extend(batch['intent_label'].cpu().tolist())

            for i in range(labels.size(0)):
                ref_ids = labels[i].cpu().tolist()
                hyp_ids = slot_ids[i].tolist()
                valid_positions = [idx for idx, rid in enumerate(ref_ids) if rid != -100]
                tokens = tokenizer.convert_ids_to_tokens(batch['input_ids'][i].cpu().tolist())
                ref = [(tokens[pos], lang.id2slot[ref_ids[pos]]) for pos in valid_positions]
                hyp = [(tokens[pos], lang.id2slot[hyp_ids[pos]]) for pos in valid_positions]
                all_refs.append(ref)
                all_hyps.append(hyp)

    slot_results = evaluate(all_refs, all_hyps)
    intent_acc = accuracy_score(intent_gts, intent_preds)
    return slot_results, {'accuracy': intent_acc}


def start_training(args):
    # BERT training setup
    train_loader, dev_loader, test_loader, lang, tokenizer = load_dataloader(
        args.data_dir,
        batch_size=args.batch_size,
        max_len=args.max_len,
        model_name=args.bert_model
    )
    model = BertForATISMultiTask.from_pretrained(
        args.bert_model,
        num_intent_labels=len(lang.intent2id),
        num_slot_labels=len(lang.slot2id)
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    best_f1, patience = 0.0, args.patience
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} Training", leave=False)
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        avg_train = train_loss / len(train_loader)

        # Validation
        slot_dev, intent_dev = evaluate_bert_epoch(dev_loader, model, tokenizer, lang)
        f1 = slot_dev['total']['f']
        print(f"Epoch {epoch}: TrainLoss={avg_train:.4f}, Dev Slot F1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1; patience = args.patience
            torch.save(model.state_dict(), 'best_bert.pt')
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping.")
                break

    # Test
    model.load_state_dict(torch.load('best_bert.pt'))
    slot_test, intent_test = evaluate_bert_epoch(test_loader, model, tokenizer, lang)
    print("Test Slot F1:", slot_test['total']['f'])
    print("Test Intent Acc:", intent_test['accuracy'])

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../dataset/ATIS')
    parser.add_argument('--bert_model', type=str, default='bert-base-uncased')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--train', action='store_true', help='Run training')
    return parser.parse_args()
