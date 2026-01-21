import torch
import torch.nn as nn

class Decoder(nn.Module):
    """
    Decoder for the seq2seq model.
    It generates the target sequence using the encoder outputs and attention mechanism.
    #EXAMPLES:
        vocab_size = 5000
        embed_dim = 256
        hidden_dim = 512
        attention = Attention(hidden_dim)
        decoder = Decoder(vocab_size, embed_dim, hidden_dim, attention)
        x = torch.tensor([1])  # Example input token
        hidden = torch.randn(1, 1, hidden_dim)  # Example hidden state
        encoder_outputs = torch.randn(1, 10, hidden_dim)  # Example encoder outputs
        prediction, new_hidden = decoder(x, hidden, encoder_outputs)
        print(prediction.shape)  # torch.Size([1, vocab_size])
        print(new_hidden.shape)  # torch.Size([1, 1, hidden_dim])
    """
    def __init__(self, vocab_size, embed_dim,hidden_dim, attention):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim + hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim *2, vocab_size)
        self.attention = attention
    def forward(self, x, hidden, encoder_outputs):
        x = x.unsqueeze(1)  # (batch_size, 1)
        embed = self.embedding(x) # (batch_size, 1, embed_dim)

        attn_weights = self.attention(hidden, encoder_outputs)  # (batch_size, seq_len)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # (batch_size, 1, hidden_dim)

        rnn_input = torch.cat((embed, context), dim=2)  # (batch_size, 1, embed_dim + hidden_dim)
        output, hidden = self.rnn(rnn_input, hidden) 

        prediction = self.fc(torch.cat((output.squeeze(1), context.squeeze(1)), dim=1))  # (batch_size, vocab_size)
        return prediction, hidden