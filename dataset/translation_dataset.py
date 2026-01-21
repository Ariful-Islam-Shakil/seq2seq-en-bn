import torch
from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    """
    Docstring for TranslationDataset.
    A dataset class for handling source and target sentences for translation tasks.
    It stores pairs of source and target sentences and provides methods to
    access them.
    #EXAMPLES:
        src_sentences = [[1, 2, 3], [4, 5, 6]]
        tgt_sentences = [[7, 8, 9], [10, 11, 12]]
        dataset = TranslationDataset(src_sentences, tgt_sentences)
        print(len(dataset))  # 2
        print(dataset[0])    # {'src': tensor([1, 2, 3]), 'tgt': tensor([7, 8, 9])}
    """
    def __init__(self, src_sentences, tgt_sentences):
        self.src = src_sentences
        self.tgt = tgt_sentences
    
    def __len__(self):
        return len(self.src)
    
    def __getitem__(self, idx):
        return {
            "src": torch.tensor(self.src[idx]),
            "tgt": torch.tensor(self.tgt[idx])
        }
