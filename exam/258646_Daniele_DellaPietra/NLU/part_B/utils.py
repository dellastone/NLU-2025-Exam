# Add functions or classes used for data loading and preprocessing
# utils.py
import os
import json
from collections import Counter
import torch
import torch.utils.data as data
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from transformers import BertTokenizerFast, BertPreTrainedModel, BertModel,  get_linear_schedule_with_warmup
import torch.nn as nn

PAD_TOKEN = 0
UNK_TOKEN = 1
def load_data(path):
    '''
        input: path/to/data
        output: json 
    '''
    dataset = []
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset

class Lang():
    def __init__(self, words, intents, slots, cutoff=0):
        self.word2id = self.w2id(words, cutoff=cutoff, unk=True)
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        self.id2word = {v:k for k, v in self.word2id.items()}
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}
        
    def w2id(self, elements, cutoff=None, unk=True):
        vocab = {'pad': PAD_TOKEN}
        if unk:
            vocab['unk'] = len(vocab)
        count = Counter(elements)
        for k, v in count.items():
            if v > cutoff:
                vocab[k] = len(vocab)
        return vocab
    
    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab['pad'] = PAD_TOKEN
        for elem in elements:
                vocab[elem] = len(vocab)
        return vocab

class IntentsAndSlots (data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, dataset, lang, unk='unk'):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.unk = unk
        
        for x in dataset:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])

        self.utt_ids = self.mapping_seq(self.utterances, lang.word2id)
        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt = torch.Tensor(self.utt_ids[idx])
        slots = torch.Tensor(self.slot_ids[idx])
        intent = self.intent_ids[idx]
        sample = {'utterance': utt, 'slots': slots, 'intent': intent}
        return sample
    
    # Auxiliary methods
    
    def mapping_lab(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]
    
    def mapping_seq(self, data, mapper): # Map sequences to number
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq.split():
                if x in mapper:
                    tmp_seq.append(mapper[x])
                else:
                    tmp_seq.append(mapper[self.unk])
            res.append(tmp_seq)
        return res


def collate_fn(data):
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len 
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape 
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(PAD_TOKEN)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq # We copy each sequence into the matrix
        # print(padded_seqs)
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, lengths
    # Sort data by seq lengths
    data.sort(key=lambda x: len(x['utterance']), reverse=True) 
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]
        
    # We just need one length for packed pad seq, since len(utt) == len(slots)
    src_utt, _ = merge(new_item['utterance'])
    y_slots, y_lengths = merge(new_item["slots"])
    intent = torch.LongTensor(new_item["intent"])
    
    src_utt = src_utt.to(device) # We load the Tensor on our selected device
    y_slots = y_slots.to(device)
    intent = intent.to(device)
    y_lengths = torch.LongTensor(y_lengths).to(device)
    
    new_item["utterances"] = src_utt
    new_item["intents"] = intent
    new_item["y_slots"] = y_slots
    new_item["slots_len"] = y_lengths
    return new_item


class ATISBERTDataset(data.Dataset):
    def __init__(self, examples, tokenizer: BertTokenizerFast, slot2id: dict, intent2id: dict, max_length: int = 128):
        self.tokenizer = tokenizer
        self.examples  = examples

        # ensure there’s always an 'unk' slot ID (fallback to pad if needed)
        if 'unk' not in slot2id:
            slot2id['unk'] = slot2id.get('pad', -100)
        self.slot2id   = slot2id
        self.intent2id = intent2id
        self.max_length= max_length
        self.features  = []
        self._prepare()

    def _prepare(self):
        for ex in self.examples:
            words   = ex['utterance'].split()
            slots   = ex['slots'].split()
            intent  = ex['intent']
            enc     = self.tokenizer(
                          words,
                          is_split_into_words=True,
                          padding='max_length',
                          truncation=True,
                          max_length=self.max_length,
                          return_tensors='pt'
                      )
            word_ids    = enc.word_ids(batch_index=0)
            slot_labels = []
            default_id = self.slot2id.get('unk', self.slot2id.get('pad', -100))

            for idx, widx in enumerate(word_ids):
                if widx is None:
                    slot_labels.append(-100)
                # first subtoken of a word, map its slot
                elif widx < len(slots) and (idx == 0 or widx != word_ids[idx-1]):
                    slot_labels.append(self.slot2id.get(slots[widx], default_id))
                # further subtokens → ignore
                else:
                    slot_labels.append(-100)

            intent_label = self.intent2id[intent]
            self.features.append({
                'input_ids':    enc['input_ids'].squeeze(),
                'attention_mask': enc['attention_mask'].squeeze(),
                'token_type_ids': enc['token_type_ids'].squeeze(),
                'slot_labels': torch.tensor(slot_labels, dtype=torch.long),
                'intent_label': torch.tensor(intent_label, dtype=torch.long)
            })

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]

class BertForATISMultiTask(BertPreTrainedModel):
    def __init__(self, config, num_intent_labels, num_slot_labels):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.intent_classifier = nn.Linear(config.hidden_size, num_intent_labels)
        self.slot_classifier = nn.Linear(config.hidden_size, num_slot_labels)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        slot_labels=None,
        intent_label=None
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        sequence_output = outputs.last_hidden_state  # [B, L, H]
        pooled_output = outputs.pooler_output        # [B, H]

        pooled_output = self.dropout(pooled_output)
        intent_logits = self.intent_classifier(pooled_output)  # [B, num_intents]

        sequence_output = self.dropout(sequence_output)
        slot_logits = self.slot_classifier(sequence_output)    # [B, L, num_slots]

        total_loss = None
        if slot_labels is not None and intent_label is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            slot_loss = loss_fct(
                slot_logits.view(-1, slot_logits.size(-1)),
                slot_labels.view(-1)
            )
            intent_loss = nn.CrossEntropyLoss()(intent_logits, intent_label)
            total_loss = slot_loss + intent_loss

        return {'loss': total_loss, 'slot_logits': slot_logits, 'intent_logits': intent_logits}