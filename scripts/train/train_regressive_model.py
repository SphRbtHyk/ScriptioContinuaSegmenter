"""Fine-tune XGLM-564M autoregressive model on a corpus."""
import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from torch.utils.data import Dataset
from typing import List
from tqdm import tqdm


class TextDataset(Dataset):
    def __init__(self, texts: List[str],
                 tokenizer,
                 max_length=512,
                 stride=128):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length
        self.stride = stride
        self.examples = []

        self._preprocess_data()

    def _preprocess_data(self):
        """Tokenize and chunk the texts into training examples"""
        print("Preprocessing data...")

        for text in tqdm(self.texts):
            # Tokenize the text
            encoding = self.tokenizer(
                text,
                truncation=False,  # We'll handle truncation manually
                padding=False,
                return_tensors=None,
                add_special_tokens=True
            )

            input_ids = encoding['input_ids']

            # Split into chunks with stride (overlapping sequences)
            for i in range(0, len(input_ids), self.stride):
                # Get chunk of max_length
                chunk = input_ids[i:i + self.max_length]

                # Pad if necessary (shouldn't happen with stride < max_length)
                if len(chunk) < self.max_length:
                    # For the last chunk, we can either pad or skip
                    # Here we'll pad to maintain consistent sequence length
                    padding_length = self.max_length - len(chunk)
                    chunk = chunk + \
                        [self.tokenizer.pad_token_id] * padding_length

                self.examples.append({
                    'input_ids': torch.tensor(chunk, dtype=torch.long),
                    'attention_mask': torch.tensor([1] * self.max_length,
                                                   dtype=torch.long),
                    'labels': torch.tensor(chunk, dtype=torch.long)
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def load_corpus(file_path: str, encoding='utf-8') -> List[str]:
    """Load corpus from the example text files.
    Every text is appended to single file"""
    print(f"Loading corpus from {file_path}...")
    content = []

    with open(file_path, 'r', encoding=encoding) as f:
        full_content = json.load(f)
    for _, book_content in full_content.items():
        for _, chapter_content in book_content.items():
            for _, verse_content in chapter_content.items():
                content.append(verse_content)

    print(f"Loaded {len(content)} verses from corpus")
    return content


def fine_tune_xglm(
    corpus_path: str,
    output_dir: str = "./xglm-finetuned",
    model_name: str = "facebook/xglm-564M",
    max_length: int = 512,
    stride: int = 256,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 5e-5,
    num_epochs: int = 3,
    warmup_steps: int = 100,
    save_steps: int = 100,
    logging_steps: int = 50
):
    """
    Fine-tune XGLM model on a corpus

    Args:
        corpus_path: Path to corpus file or directory
        output_dir: Directory to save the fine-tuned model
        model_name: XGLM model variant
        max_length: Maximum sequence length
        stride: Stride for overlapping sequences
        batch_size: Training batch size
        gradient_accumulation_steps: Accumulate gradients over several steps
        learning_rate: Learning rate for training
        num_epochs: Number of training epochs
        warmup_steps: Number of warmup steps
        save_steps: Save model every X steps
        logging_steps: Log training info every X steps
    """

    # Load tokenizer and model
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    print(f"Model loaded: {model_name}")
    print(f"Vocabulary size: {model.config.vocab_size}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    try:
        texts = load_corpus(corpus_path)
    except Exception:
        raise ValueError(f"Corpus path {corpus_path} not found")

    if not texts:
        raise ValueError("No text data found in corpus")

    # Create dataset
    print("Creating dataset...")
    dataset = TextDataset(
        texts=texts,
        tokenizer=tokenizer,
        max_length=max_length,
        stride=stride
    )

    print(f"Created {len(dataset)} training examples")

    # Split dataset (80% train, 20% validation)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Train dataset: {len(train_dataset)} examples")
    print(f"Validation dataset: {len(val_dataset)} examples")

    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
        pad_to_multiple_of=8  # For better performance on GPUs
    )

    # Training arguments
    training_args = TrainingArguments(
        max_steps=1500,
        eval_steps=50,   # Évaluer toutes les 50 steps
        save_steps=100,
        logging_steps=30,
        per_device_train_batch_size=2,  # Réduire le batch size
        gradient_accumulation_steps=8,  # Compenser avec plus d'accumulation
        fp16=True,  # Forcer mixed precision
        dataloader_pin_memory=False,
        dataloader_num_workers=0,  # Éviter le multiprocessing
        gradient_checkpointing=True,  # Économiser la mémoire
        no_cuda=True,  # Forcer CPU
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    # Start training
    print("\n=== Starting Fine-tuning ===")
    train_result = trainer.train()

    # Save final model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    print("\n=== Training Complete ===")
    print(f"Model saved to: {output_dir}")
    print(f"Final training loss: {metrics['train_loss']:.4f}")

    # Evaluate on validation set
    print("\n=== Final Evaluation ===")
    eval_metrics = trainer.evaluate()
    print(f"Validation loss: {eval_metrics['eval_loss']:.4f}")
    print(
        f"Validation perplexity: \
            {torch.exp(torch.tensor(eval_metrics['eval_loss'])).item():.2f}")

    return model, tokenizer


if __name__ == "__main__":
    # Example usage
    for lg in ["lat", "grc", "seals"]:
        if lg == "grc":
            DATASET = "sblgnt"
        elif lg == "lat":
            DATASET = "vulgata"
        elif lg == "seals":
            DATASET = "seals"
        else:
            raise ValueError(f"Uknown requested dataset: {DATASET}")
        corpus_path = f"./data/raw_data/{DATASET}.json"
        output_dir = f"./model/{lg}_xglm-564M-finetuned"

        # Fine-tune the model
        model, tokenizer = fine_tune_xglm(
            corpus_path=corpus_path,
            output_dir=output_dir,
            model_name="facebook/xglm-564M",
            max_length=128,
            stride=64,
            batch_size=2,
            gradient_accumulation_steps=8,
            learning_rate=1e-4,
            num_epochs=3
        )
