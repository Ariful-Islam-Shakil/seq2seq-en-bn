import torch 

from models.encoder import Encoder
from models.decoder import Decoder
from models.seq2seq import Seq2Seq
from models.attention import Attention
from tokenizer.tokenizer import Tokenizer
from tokenizer.vocab import Vocabulary
from utils.constants import SOS_IDX, EOS_IDX

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EMBED_DIM = 256
HIDDEN_DIM =  512
MAX_LEN = 50

def load_model(src_vocab_size, tgt_vocab_size):
    encoder = Encoder(src_vocab_size, EMBED_DIM, HIDDEN_DIM)
    attention = Attention(HIDDEN_DIM)
    decoder = Decoder(tgt_vocab_size, EMBED_DIM, HIDDEN_DIM, attention)

    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)
    model.load_state_dict(torch.load("checkpoints/seq2seq_en_bn.pt", map_location=DEVICE, weights_only=True))
    model.eval()
    return model

def translate_sentence(model, src_tokenizer, tgt_tokenizer, sentence):
    tokens = src_tokenizer.encode(sentence)
    src_tensor = torch.tensor(tokens).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        encoder_outputs, hidden = model.encode(src_tensor)
    input_token = torch.tensor([SOS_IDX]).to(DEVICE)
    outputs = []

    for _ in range(MAX_LEN):
        with torch.no_grad():
            output, hidden = model.decode(input_token, hidden, encoder_outputs)
        pred_token = output.argmax(1).item()
        if pred_token == EOS_IDX:
            break
        outputs.append(pred_token)
        input_token = torch.tensor([pred_token]).to(DEVICE)
    
    print(f"\n\ntoken number of output: {len(outputs)}\n\n")
    return tgt_tokenizer.decode(outputs)



def main():
    print("Loading vocab...\n")
    src_vocab = torch.load("tokenizer/src_vocab.pt", weights_only=False)
    tgt_vocab = torch.load("tokenizer/tgt_vocab.pt", weights_only=False)

    src_tokenizer = Tokenizer(src_vocab)
    tgt_tokenizer = Tokenizer(tgt_vocab)
    model = load_model(len(src_vocab), len(tgt_vocab))
    print("Model loaded successfully!\n")

    while True:
        sentence = input("\nEnter an English sentence: ")
        if sentence.lower() in ['exit', 'quit', 'q']:
            print("Exiting the translator. Goodbye!")
            break
        translation = translate_sentence(model, src_tokenizer, tgt_tokenizer, sentence)
        print(f"Translated Bengali sentence: {translation}")

if __name__ == "__main__":
    main()