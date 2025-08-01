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

        # If bidirectional, hidden features double
        feat_mult = 2 if bidirectional else 1
        self.slot_out = nn.Linear(hid_size * feat_mult, out_slot)
        self.intent_out = nn.Linear(hid_size, out_int)

    def forward(self, utterance, seq_lengths):
        # utterance: [batch_size, seq_len]
        utt_emb = self.embedding(utterance)  # [B, L, emb]
        utt_emb = self.dropout(utt_emb) if self.use_dropout else utt_emb

        packed = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy(), batch_first=True)
        packed_output, (h_n, c_n) = self.utt_encoder(packed)
        unpacked, _ = pad_packed_sequence(packed_output, batch_first=True)
        unpacked = self.dropout(unpacked) if self.use_dropout else unpacked
        last_hidden = h_n[-1, :, :]
        # # h_n: [num_layers * num_directions, B, hid_size]
        # # Take last layer's hidden states
        # if self.utt_encoder.bidirectional:
        #     # Concatenate forward and backward
        #     forward_h = h_n[-2, :, :]
        #     backward_h = h_n[-1, :, :]
        #     last_hidden = torch.cat([forward_h, backward_h], dim=1)  # [B, hid*2]
        # else:
        #     last_hidden = h_n[-1,:,:]  # [B, hid]

        # last_hidden = self.dropout(last_hidden) if self.use_dropout else last_hidden

        # Slot logits: [B, L, slots]
        slot_logits = self.slot_out(unpacked)
        # Convert to [B, slots, L] for loss
        slots = slot_logits.permute(0, 2, 1)

        # Intent logits: [B, intents]
        intent = self.intent_out(last_hidden)
        return slots, intent
