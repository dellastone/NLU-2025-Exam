import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ATISModel(nn.Module):
    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=1, bidirectional=True, use_dropout=True, dropout=0.1, pad_index=0):
        super(ATISModel, self).__init__()
        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)
        self.dropout = nn.Dropout(dropout) 
        self.use_dropout = use_dropout

        # Encoder with bidirectionality and inter-layer dropout if more than 1 layer
        self.utt_encoder = nn.LSTM(
            input_size=emb_size,
            hidden_size=hid_size,
            num_layers=n_layer,
            bidirectional=bidirectional,
            batch_first=True,
        )

        feat_mult = 2 if bidirectional else 1
        self.slot_out = nn.Linear(hid_size * feat_mult, out_slot)
        self.intent_out = nn.Linear(hid_size * feat_mult, out_int)

    def forward(self, utterance, seq_lengths):
        utt_emb = self.embedding(utterance)               # [B, L, E]
        if self.use_dropout:
            utt_emb = self.dropout(utt_emb)

        packed = pack_padded_sequence(
            utt_emb, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, (h_n, c_n) = self.utt_encoder(packed)
        unpacked, _ = pad_packed_sequence(packed_output, batch_first=True)  # [B, L, H*dirs]
        if self.use_dropout:
            unpacked = self.dropout(unpacked)

        if self.utt_encoder.bidirectional:
            last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)  # [B, 2H]
        else:
            last_hidden = h_n[-1]                               # [B, H]
        if self.use_dropout:
            last_hidden = self.dropout(last_hidden)

        slot_logits = self.slot_out(unpacked)   # [B, L, slots]
        slots = slot_logits.permute(0, 2, 1)    # [B, slots, L] for CE loss

        intent = self.intent_out(last_hidden)   # [B, intents]
        return slots, intent
