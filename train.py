import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
import os

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
EPOCHS = 32  # Increased default epochs for demonstration
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
CHECKPOINT_PATH = "checkpoints"
DRIVE_PATH = "/content/drive/MyDrive/Seq2Seq_model/checkpoints"


# -------------------------
# Data Loader
# -------------------------
def load_data():
    train_src, train_tgt = torch.load("data/processed/train.pt", weights_only=False)

    # train_src = train_src[:10000]
    # train_tgt = train_tgt[:10000]

    return train_src, train_tgt


# -------------------------
# Checkpoint Functions
# -------------------------
def save_checkpoint(state, epoch, filepath =CHECKPOINT_PATH, drive_path = None):
    filename = f"seq2seq_en_bn_{epoch}.pt"
    print(f"=> Saving checkpoint at {filepath}/{filename}")
    torch.save(state, f"{filepath}/{filename}")
    if drive_path:
        print(f"=> Saving checkpoint at {drive_path}/{filename}")
        torch.save(state, f"{drive_path}/{filename}")

def load_checkpoint(checkpoint_path, model, optimizer):
    print(f"=> Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=DEVICE)
    
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        if 'optimizer' in checkpoint and optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        return checkpoint.get('epoch', 0)
    else:
        # Backward compatibility for old state_dict only checkpoints
        print("=> Warning: Old checkpoint format detected (state_dict only). Optimizer state and epoch count not restored.")
        model.load_state_dict(checkpoint)
        return 0

# -------------------------
# Training Pipeline
# -------------------------
def main():
    print("Loading data...")
    train_src, train_tgt = load_data()
    print(f"Train samples: {len(train_src)}")
    print(f"Device: {DEVICE}\n\n")

    dataset = TranslationDataset(train_src, train_tgt)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print("Loading vocab...")
    src_vocab = torch.load("tokenizer/src_vocab.pt", weights_only=False)
    tgt_vocab = torch.load("tokenizer/tgt_vocab.pt", weights_only=False)

    SRC_VOCAB_SIZE = len(src_vocab)
    TGT_VOCAB_SIZE = len(tgt_vocab)

    print(f"SRC_VOCAB_SIZE: {SRC_VOCAB_SIZE}")
    print(f"TGT_VOCAB_SIZE: {TGT_VOCAB_SIZE}")

    print("Building model...")
    encoder = Encoder(SRC_VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM)
    attention = Attention(HIDDEN_DIM)
    decoder = Decoder(TGT_VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, attention)

    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    trainer = Trainer(model, optimizer, criterion)

    parser = argparse.ArgumentParser(description="Train Seq2Seq model")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()

    start_epoch = 0
    if args.resume and os.path.exists(CHECKPOINT_PATH):
        start_epoch = load_checkpoint(CHECKPOINT_PATH, model, optimizer)
        print(f"Resuming from epoch {start_epoch + 1}")

    print("Training started...\n")

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        epoch_loss = 0.0

        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch [{epoch+1}/{EPOCHS}]",
            leave=True
        )

        for batch in progress_bar:
            src = batch["src"].to(DEVICE)
            tgt = batch["tgt"].to(DEVICE)

            loss = trainer.train_step(src, tgt)
            epoch_loss += loss

            progress_bar.set_postfix(loss=f"{loss:.4f}")

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Average Loss: {avg_loss:.4f}")

        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1
        }
        save_checkpoint(checkpoint)

    print("\nTraining complete. Model saved.")


if __name__ == "__main__":
    main()
