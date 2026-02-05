"""
Example training script for fine-tuning ML models.
Supports PyTorch and Hugging Face Transformers.
"""

import argparse
import json

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--model-name", type=str, default="bert-base-uncased", help="Base model name")
    parser.add_argument("--output-dir", type=str, default="outputs/model", help="Output directory")
    parser.add_argument("--validation-only", action="store_true", help="Run quick validation only")
    parser.add_argument("--config", type=str, help="Path to config file")
    return parser.parse_args()


def load_dataset(validation_only=False):
    """
    Load your dataset here.
    For validation, return a small subset.
    """
    # TODO: Replace with your actual dataset loading logic
    from datasets import load_dataset
    
    if validation_only:
        dataset = load_dataset("imdb", split="train[:100]")  # Small sample
    else:
        dataset = load_dataset("imdb", split="train")
    
    return dataset


def main():
    args = parse_args()
    
    # Load config if provided
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
        # Merge config with args
        for key, value in config.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)
    
    print(f"Starting fine-tuning with args: {args}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2  # Adjust based on your task
    )
    
    # Load dataset
    dataset = load_dataset(validation_only=args.validation_only)
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Split into train/eval
    if args.validation_only:
        train_dataset = tokenized_dataset
        eval_dataset = tokenized_dataset
    else:
        split_dataset = tokenized_dataset.train_test_split(test_size=0.1)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save model
    print(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Save training metrics
    metrics = trainer.evaluate()
    with open(f"{args.output_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print("Training completed!")
    print(f"Final metrics: {metrics}")


if __name__ == "__main__":
    main()
