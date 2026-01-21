# Seq2Seq English-Bengali Neural Machine Translation

## Project Description
This project implements a sequence-to-sequence (Seq2Seq) neural machine translation model with attention for translating English sentences to Bengali. The architecture is based on an encoder-decoder framework using PyTorch, and includes custom tokenization, vocabulary building, and data preprocessing pipelines. The project supports training, inference, and interactive translation.

## Features
- Custom data preprocessing and tokenization
- Encoder-Decoder architecture with attention mechanism
- Training pipeline with progress tracking
- Interactive translation from English to Bengali
- Modular codebase for easy extension

## Directory Structure
- `main.py`: (Entry point, currently empty)
- `preprocess.py`: Data preprocessing and vocabulary building
- `train.py`: Model training script
- `translate.py`: Interactive translation script
- `models/`: Encoder, Decoder, Attention, and Seq2Seq model definitions
- `tokenizer/`: Tokenizer and vocabulary classes
- `dataset/`: Dataset class for translation pairs
- `training/`: Training utilities
- `utils/`: Constants and device utilities
- `data/`: Raw and processed data
- `checkpoints/`: Saved model checkpoints
- `inference/`: Translator class for inference

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repo-url>
cd seq2seq-en-bn
```

### 2. Install Dependencies
Create a virtual environment (recommended) and install required packages:
```bash
python3 -m venv venv
source venv/bin/activate
pip install torch pandas scikit-learn tqdm
```

### 3. Prepare Data
- Place your parallel English and Bengali text files as `1_Eng.txt` and `1_Bengali.txt` in the project root.
- If you already have `en_bn.csv` file, make comment of `making csv file` in `preprocess.py` file.
- Run the preprocessing script to generate processed datasets and vocabularies:
```bash
python preprocess.py
```

### 4. Train the Model
Train the Seq2Seq model using:
```bash
python train.py
```
Model checkpoints will be saved in the `checkpoints/` directory.

### 5. Translate Sentences
Run the interactive translation script:
```bash
python translate.py
```
Enter English sentences to receive Bengali translations. Type `exit`, `quit`, or `q` to stop.

## File Overview
- `preprocess.py`: Reads raw data, cleans, tokenizes, builds vocab, splits, and saves processed data.
- `train.py`: Loads processed data, builds model, trains, and saves checkpoints.
- `translate.py`: Loads model and vocab, provides interactive translation.
- `models/`: Contains `Encoder`, `Decoder`, `Attention`, and `Seq2Seq` classes.
- `tokenizer/`: Contains `Tokenizer` and `Vocabulary` classes.
- `dataset/translation_dataset.py`: PyTorch Dataset for translation pairs.

## Notes
- Adjust hyperparameters in `train.py` as needed (embedding size, hidden size, batch size, epochs, etc.).
- The project currently uses PyTorch and standard Python libraries.
- For GPU training, ensure CUDA is available and PyTorch is installed with CUDA support.

## Example Usage
```
$ python translate.py
Loading vocab...
Model loaded successfully!

Enter an English sentence: I love you
Translated Bengali sentence: আমি তোমাকে ভালোবাসি
```

## License
MIT License
