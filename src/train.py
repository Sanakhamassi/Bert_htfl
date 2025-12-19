from transformers import Trainer, TrainingArguments, DataCollatorForTokenClassification
from seqeval.metrics import f1_score, precision_score, recall_score
import numpy as np

from dataset import load_tsv
from model import load_model
from tokenizer_utils import tokenize_and_align_labels, tokenizer, id2label
# definition of hyperparameters 
lr = 2e-5
num_epochs= 10
#   Adapted for Trainer - takes EvalPrediction object
def compute_metrics(eval_pred):
    predictions, labels = eval_pred  # Unpack the EvalPrediction object
    preds = np.argmax(predictions, axis=-1)
    
    # Remove padding tokens (-100) and convert to label strings
    true_labels = [
        [id2label[l] for l in label if l != -100] 
        for label in labels
    ]
    pred_labels = [
        [id2label[p] for (p, l) in zip(pred, label) if l != -100] 
        for pred, label in zip(preds, labels)
    ]
    
    return {
        "precision": precision_score(true_labels, pred_labels),
        "recall": recall_score(true_labels, pred_labels),
        "f1": f1_score(true_labels, pred_labels),
    }


# Load datasets
train_ds = load_tsv("../data/processed/train.tsv")
val_ds   = load_tsv("../data/processed/val.tsv")

# Tokenize and align labels
train_ds = train_ds.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=train_ds.column_names
)

val_ds = val_ds.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=val_ds.column_names
)

# Load model
model = load_model()
# to make each sample in the batch to the same size
# Data collator for token classification
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# Training arguments
args = TrainingArguments(
    output_dir="../model_checkpoints/biobert_htfl",
    eval_strategy="epoch",   #per epoch       
    save_strategy="epoch",
    learning_rate=lr,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1" 
)

# Trainer 
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics  
)

trainer.train()