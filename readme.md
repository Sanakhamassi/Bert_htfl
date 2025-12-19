# BioBERT Fine-Tuning for Heart Failure Term Extraction on (ACTER) dataset

## Project Overview
This project fine-tunes BioBERT for Automatic Term Extraction (ATE) in the heart failure domain using the ACTER dataset. The task is formulated as a token-level sequence labeling problem to identify domain-specific terms in biomedical text.

## Task Description
- Task type: Token Classification (Sequence Labeling)
- Objective: Automatically extract heart failure–related terms
- Labeling scheme:
  - O: Outside a term
  - I: Inside a term

## Model
- Base model: dmis-lab/biobert-base-cased-v1.1
- Architecture: AutoModelForTokenClassification
- Number of labels: 2 (O, I)

BioBERT is pre-trained on large-scale biomedical corpora, making it suitable for medical NLP tasks such as term extraction.

## Dataset
- Dataset: ACTER (Automatic Term Extraction Dataset)
- Domain: Heart Failure
- Format: TSV (token and label per line)

Example:
heart   I
failure I
is      O
a       O
disease O

Sentences are separated by blank lines.

## Training Pipeline
1. Load TSV files into Hugging Face Dataset objects.
2. Tokenize input using BioBERT tokenizer and align labels with subword tokens.
3. Fine-tune BioBERT using Hugging Face Trainer.
4. Select the best model based on validation loss.

# Results
The model was trained for 10 epochs on the ACTER heart failure dataset. Evaluation was performed at the end of each epoch on the validation set.

### Best Validation Performance
- Precision: **0.862**
- Recall: **0.899**
- F1-score: **0.879**
- Validation loss: **0.195**

The best F1-score was achieved at epoch 6–10, indicating stable convergence. Training loss steadily decreased, demonstrating effective fine-tuning of BioBERT for the automatic term extraction task.

### Training Details
- Total training time: ~15.7 minutes
- Final training loss: 0.026
- Hardware: Apple Silicon (MPS backend)

## Model checkpoints
The fine-tuned model will be saved in:
experiments/biobert_htfl/




