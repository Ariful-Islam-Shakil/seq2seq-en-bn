import torch
from utils.constants import SOS_TOKEN, EOS_TOKEN
class Translator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    def translate(self, sentence, max_len = 50):
        self.model.eval()
        with torch.no_grad():
            # Encode the input sentence
            input_indices = self.tokenizer.encode(sentence)
            input_tensor = torch.tensor(input_indices).unsqueeze(0)  # (1, seq_len)
            encoder_outputs, hidden = self.model.encoder(input_tensor)

            # Initialize the target sequence with SOS token
            target_indices = [self.tokenizer.vocab.token2idx[SOS_TOKEN]]
            for _ in range(max_len):
                target_tensor = torch.tensor([target_indices[-1]])  # (1)
                prediction, hidden = self.model.decoder(target_tensor, hidden, encoder_outputs)
                predicted_id = prediction.argmax(1).item()
                if predicted_id == self.tokenizer.vocab.token2idx[EOS_TOKEN]:
                    break
                target_indices.append(predicted_id)

            translated_sentence = self.tokenizer.decode(target_indices[1:])  # Exclude SOS token
            return translated_sentence