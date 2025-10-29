import torch
import torch.nn as nn
from decoder import PositionalEncoding, EncoderLayer, Decoder

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            Decoder(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        # Padding masks
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)   # [B, 1, 1, src_len]
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)   # [B, 1, 1, tgt_len]

        # Causal (no-peek) mask
        seq_length = tgt.size(1)
        nopeak_mask = torch.tril(torch.ones((1, seq_length, seq_length), device=tgt.device)).bool()

        # Combine
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt, return_attention=False):
        """
        Forward pass through the transformer.

        Args:
            src (Tensor): [batch_size, src_len]
            tgt (Tensor): [batch_size, tgt_len]
            return_attention (bool): If True, returns (output, attn_weights)

        Returns:
            Tensor: Output logits [batch_size, tgt_len, vocab_size]
            Optional[List[Tensor]]: attention maps from each decoder layer
        """
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        # Encoder
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        # Decoder (collect attention maps)
        dec_output = tgt_embedded
        all_attn_weights = []
        for dec_layer in self.decoder_layers:
            dec_output, attn_weights = dec_layer(dec_output, enc_output, src_mask, tgt_mask, return_attention=True)
            if return_attention:
                all_attn_weights.append(attn_weights)

        output = self.fc(dec_output)

        if return_attention:
            return output, all_attn_weights
        return output
