import torch
import torch.nn as nn

class Seq2Seq(nn.Module):
    """
    Docstring for Seq2Seq
    A sequence-to-sequence model that combines an encoder and a decoder.
    It processes input sequences and generates output sequences.
    #EXAMPLES:  
        encoder = Encoder(vocab_size=5000, embed_dim=256, hidden_dim=512)
        attention = Attention(hidden_dim=512)
        decoder = Decoder(vocab_size=5000, embed_dim=256, hidden_dim=512, attention=attention)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = Seq2Seq(encoder, decoder, device)
        src = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])  # Example source batch
        tgt = torch.tensor([[1, 9, 10, 2], [1, 11, 12, 2]])  # Example target batch
        outputs = model(src, tgt)
        print(outputs.shape)  # torch.Size([2, 4, 5000])
    """ 
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    
    def encode(self, src):
        return self.encoder(src)
    
    def decode(self, input, hidden, encoder_outputs):
        return self.decoder(input, hidden, encoder_outputs)

    def forward(self, src, tgt, teacher_forching_ratio = 0.5):
        """
        Docstring for forward
        Processes the source and target sequences to generate outputs.
        Args:
            src (torch.Tensor): The source sequences of shape (batch_size, src_len).
            tgt (torch.Tensor): The target sequences of shape (batch_size, tgt_len).
            teacher_forching_ratio (float): The ratio for teacher forcing.
        Returns:
            torch.Tensor: The output sequences of shape (batch_size, tgt_len, vocab_size).
        """
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(batch_size, tgt_len, vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src)
        input = tgt[:, 0]
        for t in range(1, tgt_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs) 
            outputs[:, t] = output
            teacher_force = torch.rand(1).item() < teacher_forching_ratio
            input = tgt[:, t] if teacher_force else output.argmax(1)
        
        return outputs

    