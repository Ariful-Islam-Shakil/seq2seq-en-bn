import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from dataset.translation_dataset import TranslationDataset
from models.encoder import Encoder
from models.decoder import Decoder
from models.attention import Attention
from models.seq2seq import Seq2Seq
from training.trainer import Trainer
from utils.constants import PAD_IDX
from tokenizer.vocab import Vocabulary


# -------------------------
# Configuration
# -------------------------
EMBED_DIM = 256
HIDDEN_DIM = 512
BATCH_SIZE = 32
EPOCHS = 2
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# Dummy loader (replace with real data loading)
# -------------------------
def load_data():
    """
    Expected output:
    train_src: List[List[int]]
    train_tgt: List[List[int]]
    """
    train_data = torch.load("data/processed/train.pt", weights_only=False)
    train_src, train_tgt = train_data
    
    return train_src, train_tgt


# -------------------------
# Training Pipeline
# -------------------------
def main():
    print("Loading data...")
    train_src, train_tgt = load_data()

    dataset = TranslationDataset(train_src, train_tgt)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print("Loading vocab...")
    src_vocab = torch.load("tokenizer/src_vocab.pt", weights_only=False)
    tgt_vocab = torch.load("tokenizer/tgt_vocab.pt", weights_only=False)

    SRC_VOCAB_SIZE = len(src_vocab)
    TGT_VOCAB_SIZE = len(tgt_vocab)

    print("Building model...")
    encoder = Encoder(SRC_VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM)
    attention = Attention(HIDDEN_DIM)
    decoder = Decoder(TGT_VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, attention)

    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    trainer = Trainer(model, optimizer, criterion)

    print("Training started...")
    for epoch in range(EPOCHS):
        epoch_loss = 0

        model.train()
        for batch in dataloader:
            src = batch["src"].to(DEVICE)
            tgt = batch["tgt"].to(DEVICE)

            loss = trainer.train_step(src, tgt)
            epoch_loss += loss

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f}")

        torch.save(model.state_dict(), "checkpoints/seq2seq_en_bn.pt")

    print("Training complete. Model saved.")


if __name__ == "__main__":
    main()
