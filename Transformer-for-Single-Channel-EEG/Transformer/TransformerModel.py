import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(torch.nn.Module):
    """
    This model replaces the original CompactCNN with a Transformer model.
    It is designed to classify 1D EEG signals for the purpose of driver drowsiness recognition.

    Parameters:

    classes      : number of classes to classify, the default number is 2 corresponding to the 'alert' and 'drowsy' labels.
    input_dim    : the dimension of each input vector
    nhead        : the number of heads in the multiheadattention models
    nhid         : the dimension of the feedforward network model
    nlayers      : the number of transformer encoder layers in the model
    dropout      : the dropout value (default=0.5)
    sampleLength : the length of the 1D EEG signal. The default value is 384, which is 3s signal with sampling rate of 128Hz.

    """

    def __init__(self, classes=2, input_dim=64, nhead=1, nhid=32, nlayers=2, dropout=0.5, sampleLength=384):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.nhid = nhid
        self.pos_encoder = PositionalEncoding(input_dim, dropout)
        encoder_layers = TransformerEncoderLayer(input_dim, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        #self.encoder = torch.nn.Linear(sampleLength, input_dim)
        self.encoder = torch.nn.Linear(384, input_dim)

        self.decoder = torch.nn.Linear(input_dim, classes)
        self.softmax = torch.nn.LogSoftmax(dim=1)
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    # def forward(self, src):
    #     if self.src_mask is None or self.src_mask.size(0) != len(src):
    #         device = src.device
    #         mask = self._generate_square_subsequent_mask(len(src)).to(device)
    #         self.src_mask = mask
    #
    #     src = self.encoder(src) * math.sqrt(self.nhid)
    #     src = self.pos_encoder(src)
    #     output = self.transformer_encoder(src, self.src_mask)
    #     output = self.decoder(output)
    #     output = self.softmax(output)
    #
    #     return output

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.nhid)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = output.mean(dim=1)  # Take the mean of the sequence dimension
        output = self.decoder(output)
        output = self.softmax(output)

        return output
