class Tokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
    
    def tokenize(self, sentence):
        return sentence.strip().split()
    
    def encode(self, sentence):
        tokens = self.tokenize(sentence)
        return [self.vocab.token2idx.get(t, 3) for t in tokens]  # 3 is the UNK_ID
    
    def decode(self, indices):
        return " ".join([self.vocab.idx2token[i] for i in indices])

    
