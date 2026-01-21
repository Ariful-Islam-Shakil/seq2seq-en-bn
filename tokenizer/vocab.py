class Vocabulary:
    """A vocabulary class for tokenizing text.
    It maintains a mapping between tokens and their corresponding indices,
    as well as the frequency of each token.
    It also includes special tokens for padding, start-of-sequence,
    end-of-sequence, and unknown tokens.
    #EXAMPLES:
        vocab = Vocabulary()
        vocab.add_token("hello")
        vocab.add_token("world")
        print(vocab.token2idx)  # {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3, 'hello': 4, 'world': 5}
        print(vocab.idx2token)  # {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>', 4: 'hello', 5: 'world'}
        print(vocab.freq)       # {'<PAD>': 1, '<SOS>': 1, '<EOS>': 1, '<UNK>': 1, 'hello': 1, 'world': 1}
    """

    def __init__(self):
        self.token2idx = {}
        self.idx2token = {}
        self.freq = {}
        self.add_special_tokens()
    
    def add_special_tokens(self):
        from utils.constants import PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN
        for token in [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]:
            self.add_token(token)
    
    def add_token(self, token):
        if token not in self.token2idx:
            idx = len(self.token2idx)
            self.token2idx[token] = idx
            self.idx2token[idx] = token
            self.freq[token] = 1
        else:
            self.freq[token] += 1

    def __len__(self):
        return len(self.token2idx)