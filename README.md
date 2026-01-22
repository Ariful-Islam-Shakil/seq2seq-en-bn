# Seq2Seq English-Bengali Neural Machine Translation

## Project Description
This project implements a sequence-to-sequence (Seq2Seq) neural machine translation model with attention for translating English sentences to Bengali. The architecture is based on an encoder-decoder framework using PyTorch, and includes custom tokenization, vocabulary building, and data preprocessing pipelines.

## Features
- **Encoder-Decoder with Attention**: High-performance architecture for translation tasks.
- **Resumable Training**: Support for resuming from the latest or a specific checkpoint.
- **Smart Checkpoint Management**: Automatically keeps storage clean by deleting older checkpoints (retains $n$ and $n-1$).
- **Dual Storage Saving**: Saves checkpoints to both local storage and Google Drive (optional).
- **Custom Tokenization**: Tailored cleaning and tokenization for English and Bengali.
- **Interactive Translation**: CLI tool for real-time model testing.

## Directory Structure
- `preprocess.py`: Data cleaning, vocabulary building, and dataset preparation.
- `train.py`: Model training script with resume and checkpoint management logic.
- `translate.py`: Interactive translation script for testing.
- `models/`: Neural network architecture definitions (`Encoder`, `Decoder`, `Attention`, `Seq2Seq`).
- `tokenizer/`: Tokenizer and vocabulary management logic.
- `dataset/`: Custom PyTorch `Dataset` for translation pairs.
- `training/`: Core training logic and `Trainer` class.
- `checkpoints/`: Local directory for model weights.

## Setup Instructions

### 1. Clone & Environment
```bash
git clone <repo-url>
cd seq2seq-en-bn
python3 -m venv venv
source venv/bin/activate
pip install torch pandas scikit-learn tqdm
```

### 2. Prepare Data
The preprocessing script handles CSV generation and data cleaning. You can specify the number of samples to process in `preprocess.py`.
```bash
# Processes raw text and saves to data/processed/
python preprocess.py
```

### 3. Train the Model
The training script is highly configurable:

- **Start from scratch:**
  ```bash
  python train.py
  ```
- **Resume from latest checkpoint:**
  ```bash
  python train.py --resume
  ```
- **Resume from a specific epoch (e.g., Epoch 5):**
  ```bash
  python train.py --epoch 5
  ```

*Note: The script automatically deletes the $(n-2)$-th checkpoint to save space.*

### 4. Translate Sentences
Test your model interactively:
```bash
python translate.py
```
Type `exit` or `q` to quit the session.

## File Overview
- `preprocess.py`: Cleans text, builds vocabularies (`src_vocab.pt`, `tgt_vocab.pt`), and splits data.
- `train.py`: Manages the training loop, loss calculation, and checkpointing.
- `models/seq2seq.py`: Wrapper for the encoder-decoder interaction.
- `training/trainer.py`: Handles individual training steps and optimization.

## Advanced Configuration
Hyperparameters like `EMBED_DIM`, `HIDDEN_DIM`, `BATCH_SIZE`, and `LR` can be adjusted directly in `train.py`. The model supports CUDA, MPS (for Mac), and CPU automatically.

## Example Usage
```text
$ python translate.py
Loading vocab...
Model loaded successfully!

Enter an English sentence: How are you?
Translated Bengali sentence: আপনি কেমন আছেন?
```

## License
MIT License
