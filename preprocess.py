import pandas as pd
import torch
import csv
from sklearn.model_selection import train_test_split

from tokenizer.vocab import Vocabulary
from tokenizer.tokenizer import Tokenizer
from utils.constants import SOS_IDX, EOS_IDX, PAD_IDX


MAX_LEN = 50
TEST_SIZE = 0.1
VAL_SIZE = 0.1

# -------------------------
# CSV Creation from Text Files
# -------------------------

def make_csv_from_texts(eng_path, bng_path, csv_path, num_sentences=100000):
    with open(eng_path, 'r', encoding='utf-8') as f_eng, open(bng_path, 'r', encoding='utf-8') as f_bng:
        eng_lines = [line.strip() for line in f_eng.readlines()]
        bng_lines = [line.strip() for line in f_bng.readlines()]
    pairs = list(zip(eng_lines, bng_lines))#[:num_sentences]
    
    num_samples = len(pairs)
    print(f"\nTotal pairs: {num_samples}\n")
    chunks = [pairs[i:i + num_sentences] for i in range(0, num_samples, num_sentences)]
    print(f"Number of chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks):
        csv_path = f"data/raw/en_bn_{i}.csv"
        with open(csv_path, 'w', encoding='utf-8', newline='') as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=['eng', 'bng'])
            writer.writeheader()
            for eng, bng in chunk:
                writer.writerow({'eng': eng, 'bng': bng})
        print(f"\nChunk {i+1} saved to {csv_path}\n")
# -------------------------
# Text Cleaning
# -------------------------
def clean_text(text):
    text = str(text).strip()
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text


# -------------------------
# Padding
# -------------------------
def pad_sequence(seq, max_len):
    seq = seq[:max_len]
    return seq + [PAD_IDX] * (max_len - len(seq))


# -------------------------
# Build Dataset
# -------------------------
def build_sequences(df, src_tokenizer, tgt_tokenizer):
    src_sequences = []
    tgt_sequences = []

    for _, row in df.iterrows():
        src = clean_text(row["english"])
        tgt = clean_text(row["bangla"])

        src_ids = src_tokenizer.encode(src)
        tgt_ids = tgt_tokenizer.encode(tgt)

        src_ids = [SOS_IDX] + src_ids + [EOS_IDX]
        tgt_ids = [SOS_IDX] + tgt_ids + [EOS_IDX]

        src_ids = pad_sequence(src_ids, MAX_LEN)
        tgt_ids = pad_sequence(tgt_ids, MAX_LEN)

        src_sequences.append(src_ids)
        tgt_sequences.append(tgt_ids)

    return src_sequences, tgt_sequences


# -------------------------
# Main Pipeline
# -------------------------
def preprocess_data(csv_path):
    print("Loading CSV...")
    df = pd.read_csv(csv_path)
    df.columns = ["english", "bangla"]
    df = df.head(20000)
    df.to_csv("data/raw/en_bn.csv", index=False)

    df["english"] = df["english"].apply(clean_text)
    df["bangla"] = df["bangla"].apply(clean_text)

    print("Building vocabularies...")
    src_vocab = Vocabulary()
    tgt_vocab = Vocabulary()

    for text in df["english"]:
        for token in text.split():
            src_vocab.add_token(token)

    for text in df["bangla"]:
        for token in text.split():
            tgt_vocab.add_token(token)

    src_tokenizer = Tokenizer(src_vocab)
    tgt_tokenizer = Tokenizer(tgt_vocab)

    print("Splitting dataset...")
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=VAL_SIZE, random_state=42)

    print("Encoding datasets...")
    train_src, train_tgt = build_sequences(train_df, src_tokenizer, tgt_tokenizer)
    val_src, val_tgt = build_sequences(val_df, src_tokenizer, tgt_tokenizer)
    test_src, test_tgt = build_sequences(test_df, src_tokenizer, tgt_tokenizer)

    print("Saving processed data...")
    torch.save((train_src, train_tgt), "data/processed/train.pt")
    torch.save((val_src, val_tgt), "data/processed/val.pt")
    torch.save((test_src, test_tgt), "data/processed/test.pt")

    torch.save(src_vocab, "tokenizer/src_vocab.pt")
    torch.save(tgt_vocab, "tokenizer/tgt_vocab.pt")

    print("Preprocessing complete ✅")
    print(f"Train samples: {len(train_src)}")
    print(f"Vocab sizes → EN: {len(src_vocab)}, BN: {len(tgt_vocab)}")

if __name__ == "__main__":
    # make_csv_from_texts(
    #     eng_path="1_Eng.txt",
    #     bng_path="1_Bengali.txt",
    #     csv_path="/Users/mdarifulislamshakil/MyProjects/seq2seq-en-bn/data/raw/en_bn.csv",
    #     num_sentences=2000000
    # )
    # print("\n\nCSV created successfully!\n\n")
    
    preprocess_data("data/raw/en_bn_0.csv")