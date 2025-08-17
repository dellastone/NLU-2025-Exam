import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertTokenizerFast, BertPreTrainedModel, BertModel, get_linear_schedule_with_warmup



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

