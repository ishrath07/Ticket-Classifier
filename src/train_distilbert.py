"""
train_distilbert.py
Fine-tunes distilbert-base-uncased for ticket classification.
Run once per label — this keeps the two models fully independent,
matching the approach used for the sklearn baseline.

Usage:
    python src/train_distilbert.py --task category
    python src/train_distilbert.py --task urgency
"""

import argparse
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

MODEL_NAME = "distilbert-base-uncased"
TRAIN_PATH = "data/processed/train.csv"
VAL_PATH = "data/processed/val.csv"
MAX_LENGTH = 128


def load_and_encode(task: str):
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)

    label_encoder = LabelEncoder()
    label_encoder.fit(train_df[task])

    train_df = train_df.copy()
    val_df = val_df.copy()
    train_df["label"] = label_encoder.transform(train_df[task])
    val_df["label"] = label_encoder.transform(val_df[task])

    train_ds = Dataset.from_pandas(train_df[["ticket_text", "label"]])
    val_ds = Dataset.from_pandas(val_df[["ticket_text", "label"]])

    return train_ds, val_ds, label_encoder


def tokenize_datasets(train_ds, val_ds, tokenizer):
    def tokenize_fn(batch):
        return tokenizer(
            batch["ticket_text"], truncation=True, padding="max_length", max_length=MAX_LENGTH
        )

    train_ds = train_ds.map(tokenize_fn, batched=True)
    val_ds = val_ds.map(tokenize_fn, batched=True)
    return train_ds, val_ds


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1_macro": f1_macro}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["category", "urgency"], required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    output_dir = f"models/distilbert_{args.task}"

    print(f"Loading data for task: {args.task}")
    train_ds, val_ds, label_encoder = load_and_encode(args.task)
    num_labels = len(label_encoder.classes_)
    print(f"Labels ({num_labels}): {list(label_encoder.classes_)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds, val_ds = tokenize_datasets(train_ds, val_ds, tokenizer)

    id2label = {i: label for i, label in enumerate(label_encoder.classes_)}
    label2id = {label: i for i, label in enumerate(label_encoder.classes_)}

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels, id2label=id2label, label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir=f"{output_dir}_checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=20,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    print(f"\nFine-tuning DistilBERT for: {args.task}")
    trainer.train()

    print("\nFinal validation metrics:")
    metrics = trainer.evaluate()
    print(metrics)

    # Save final model, tokenizer, and label mapping together
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    with open(f"{output_dir}/label_mapping.json", "w") as f:
        json.dump({"id2label": id2label, "label2id": label2id}, f, indent=2)

    print(f"\nSaved fine-tuned model to {output_dir}/")


if __name__ == "__main__":
    main()
