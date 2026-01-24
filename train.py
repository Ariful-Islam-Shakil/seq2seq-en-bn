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
EPOCHS = 30  # Increased default epochs for demonstration
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
CHECKPOINT_PATH = "checkpoints"
DRIVE_PATH = "/content/drive/MyDrive/Seq2Seq_model/checkpoints"


# -------------------------
# Data Loader
# -------------------------
def load_data():
    train_src, train_tgt = torch.load("data/processed/train.pt", weights_only=False, size = None)
    if size:
        train_src = train_src[:size]
        train_tgt = train_tgt[:size]

    return train_src, train_tgt


# -------------------------
# Checkpoint Functions
# -------------------------
def delete_checkpoint(epoch, filepath=CHECKPOINT_PATH, drive_path=DRIVE_PATH):
    filename = f"seq2seq_en_bn_{epoch}.pt"
    local_file = os.path.join(filepath, filename)
    if os.path.exists(local_file):
        try:
            os.remove(local_file)
            print(f"=> Deleted old local checkpoint: {local_file}")
        except Exception as e:
            print(f"=> Error deleting local checkpoint: {e}")

    if drive_path and os.path.exists(drive_path):
        drive_file = os.path.join(drive_path, filename)
        if os.path.exists(drive_file):
            try:
                os.remove(drive_file)
                print(f"=> Deleted old drive checkpoint: {drive_file}")
            except Exception as e:
                print(f"=> Error deleting drive checkpoint: {e}")

def save_checkpoint(state, epoch, filepath=CHECKPOINT_PATH, drive_path=DRIVE_PATH):
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    
    filename = f"seq2seq_en_bn_{epoch}.pt"
    print(f"=> Saving checkpoint at {filepath}/{filename}")
    torch.save(state, f"{filepath}/{filename}")
    
    if drive_path:
        try:
            if not os.path.exists(drive_path):
                os.makedirs(drive_path)
            print(f"=> Saving checkpoint at {drive_path}/{filename}")
            torch.save(state, f"{drive_path}/{filename}")
        except Exception as e:
            print(f"=> Error saving checkpoint to drive: {e}")
            
    # Delete (n-2)th checkpoint
    if epoch > 2:
        delete_checkpoint(epoch - 2, filepath, drive_path)

def load_checkpoint(checkpoint_path, model, optimizer, epoch=1):
    file_path = f"{checkpoint_path}/seq2seq_en_bn_{epoch}.pt"
    if not os.path.exists(file_path):
        print(f"=> Error: Checkpoint file not found: {file_path}")
        return 0

    print(f"=> Loading checkpoint from {file_path}")
    checkpoint = torch.load(file_path, weights_only=False, map_location=DEVICE)
    
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
def main(data_size = None):
    print("Loading data...")
    train_src, train_tgt = load_data(data_size)
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
    parser.add_argument("--epoch", type=int, default=None, help="Specific epoch to resume from")
    args = parser.parse_args()

    start_epoch = 0
    if args.epoch is not None:
        start_epoch = load_checkpoint(DRIVE_PATH, model, optimizer, epoch=args.epoch)
        print(f"Resuming from after epoch {start_epoch}")
    elif args.resume and os.path.exists(CHECKPOINT_PATH):
        # If --resume is used without --epoch, try to find the latest
        checkpoints = [f for f in os.listdir(DRIVE_PATH) if f.startswith("seq2seq_en_bn_") and f.endswith(".pt")]
        if checkpoints:
            latest_epoch = max([int(f.split("_")[-1].split(".")[0]) for f in checkpoints])
            start_epoch = load_checkpoint(DRIVE_PATH, model, optimizer, epoch=latest_epoch)
            print(f"Resuming from after epoch {start_epoch}")
        else:
            print("=> No checkpoints found to resume from.")

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
        save_checkpoint(checkpoint, epoch+1)

    print("\nTraining complete. Model saved.")


if __name__ == "__main__":
    data_size = 1000
    main(data_size)
