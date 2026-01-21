import torch
import torch.nn as nn

class Attention(nn.Module):
    """
    Docstring for Attention
    An attention mechanism that computes attention scores between
    the decoder hidden state and encoder outputs.
    #EXAMPLES:
        attention = Attention(hidden_dim=256)
        hidden = torch.randn(1, 1, 256)  # Decoder hidden state
        encoder_outputs = torch.randn(1, 10, 256)  # Encoder outputs
        scores = attention(hidden, encoder_outputs)
        print(scores.shape)  # torch.Size([1, 10])
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim *2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        """
        Docstring for forward
        Computes attention scores.
        Args:
            hidden (torch.Tensor): The current decoder hidden state of shape (batch_size, 1, hidden_dim).
            encoder_outputs (torch.Tensor): The encoder outputs of shape (batch_size, seq_len, hidden_dim).
        Returns:
            torch.Tensor: Attention scores of shape (batch_size, seq_len).
        """
        seq_len = encoder_outputs.shape[1]
        hidden = hidden.transpose(0, 1).repeat(1, seq_len, 1) # (batch_size, seq_len, hidden_dim)

        energy = torch.tanh(self.attn(
            torch.cat((hidden, encoder_outputs), dim=2)
        ))
        scores = self.v(energy).squeeze(2)

        return torch.softmax(scores, dim=1)
        