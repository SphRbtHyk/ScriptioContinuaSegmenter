"""Train the character embedding model with quadrimodal labelings.
"""
import torch
from transformers import AutoTokenizer, CanineForTokenClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from typing import List, Tuple
from transformers import EarlyStoppingCallback


def map_labels_quadri(list_labels: list[str]) -> list[int]:
    """Map the label plan designed for the quadrimodal classification
    task into a list of integer.
    """
    quadri_labels = []
    MAPPER = {
        "I": 0,
        "E": 1,
        "B": 2,
        "S": 3
    }
    for label in list_labels:
        quadri_labels.append(MAPPER[label.strip()])
    return quadri_labels


def load_dataset(data_path: str):
    """Load the dataset as a list of tuple."""
    dataset = []
    with open(data_path, "r") as f:
        raw_lines = f.readlines()
    for line in raw_lines:
        sentence, tags_string, ground_truth = line.split("-")
        # Tags are now a string with a list of tags
        tags_list = tags_string.replace(
            "[", "").replace("]", "").replace("'", "").split(",")
        dataset.append(tuple((sentence.strip(),
                              map_labels_quadri(tags_list))))
    return dataset


class CharacterLevelDataset(Dataset):
    def __init__(self, sentences: List[str],
                 all_char_labels: List[List[int]],
                 tokenizer,
                 max_length=512):
        self.tokenizer = tokenizer
        self.sentences = sentences
        self.all_char_labels = all_char_labels
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        char_labels = self.all_char_labels[idx]

        # Tokenize the sentence
        encoding = self.tokenizer(
            sentence,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Create labels tensor aligned with tokenization
        labels = self._align_labels_with_tokens(
            sentence, char_labels, encoding)

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0)
        }

    def _align_labels_with_tokens(self, sentence: str, char_labels: List[int], encoding) -> torch.Tensor:
        """Align character labels with CANINE's tokenization"""
        tokens = self.tokenizer.tokenize(sentence)

        # Initialize labels with ignore index (-100)
        labels = torch.full(encoding['input_ids'].shape, -100)

        # Current character position in original sentence
        char_pos = 0
        # Current token position (starting after [CLS])
        token_pos = 1  # position after [CLS]

        for token in tokens:
            if token_pos >= self.max_length - 1:  # Leave room for [SEP]
                break

            if token == " ":
                # Space token - we ignore it (use -100)
                token_pos += 1
                continue

            # For most cases: 1 character = 1 token
            if char_pos < len(char_labels):
                labels[0, token_pos] = char_labels[char_pos]
                char_pos += 1
            token_pos += 1

        return labels


def prepare_data(path: str) -> List[Tuple[str, List[int]]]:
    """Load data"""
    return load_dataset(path)


def verify_alignment(sentence: str, char_labels: List[int], tokenizer):
    """Verify that character labels are correctly aligned with tokens"""
    print("\n=== Verifying Alignment ===")
    print(f"Sentence: '{sentence}'")
    print(f"Characters: {list(sentence)}")
    print(f"Char labels: {char_labels}")

    tokens = tokenizer.tokenize(sentence)
    print(f"CANINE tokens: {tokens}")

    # Test the alignment function
    dataset = CharacterLevelDataset([sentence], [char_labels], tokenizer)
    item = dataset[0]

    # Decode to see alignment
    input_ids = item['input_ids']
    labels = item['labels']

    print("Final alignment:")
    for i, (token_id, label) in enumerate(zip(input_ids, labels)):
        token = tokenizer.decode([token_id])
        if token_id not in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]:
            print(f"  Pos {i:2d}: Token '{token}' -> Label {label.item()}")


def train(LANGUAGE,
          MODEL_NAME="google/canine-s",
          NUM_LABELS=4,
          MAX_LENGTH=128):
    # Configuration

    # Dataset
    TRAINING_FILE = {
        "lat": "data/ground_truth/vulgata_quadrimodal.txt",
        "grc": "data/ground_truth/sblgnt_quadrimodal.txt",
        "seals": "data/ground_truth/seals_quadrimodal.txt"
    }
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = CanineForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label={0: "inside",
                  1: "end",
                  2: "beginning",
                  3: "single"},
        label2id={"inside": 0,
                  "end": 1,
                  "beginning": 2,
                  "single": 3}
    )
    # MAPPER = {
    #     "I": 0,
    #     "E": 1,
    #     "B": 2,
    #     "S": 3
    # }
    # Prepare your data
    training_data = prepare_data(TRAINING_FILE[LANGUAGE])
    sentences = [data[0] for data in training_data]
    all_char_labels = [data[1] for data in training_data]

    # Verify alignment for first sample
    verify_alignment(sentences[0], all_char_labels[0], tokenizer)

    # Create dataset
    dataset = CharacterLevelDataset(
        sentences, all_char_labels, tokenizer, max_length=MAX_LENGTH)

    # Split dataset in train / test
    # This is really not ideal, add some randomness in it
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])
    output_dir = f"./model/{LANGUAGE}_character-classifier-quadri"
    # LONGER TRAINING
    # # Training arguments
    training_args = TrainingArguments(
        #max_steps=3,
        output_dir=output_dir,
        overwrite_output_dir=True,

        # ===== TRAINING DURATION =====
        num_train_epochs=10,
        # Increased from 3 to 10 for more thorough training
        max_steps=10000,  # Set a high maximum steps limit

        # ===== BATCH CONFIGURATION =====
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,  # Increased: effective batch size = 8 * 4 = 32
        eval_accumulation_steps=2,  # Accumulate eval steps to save memory

        # ===== OPTIMIZATION =====
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=1000,  # Increased for longer training
        warmup_ratio=0.1,  # Alternative: 10% of total steps for warmup

        # Advanced optimization
        adam_epsilon=1e-8,
        adam_beta1=0.9,
        adam_beta2=0.999,
        max_grad_norm=1.0,  # Gradient clipping

        # ===== SCHEDULING =====
        # Options: "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"
        lr_scheduler_type="linear",
        # For cosine scheduling:
        # lr_scheduler_type="cosine",
        # learning_rate=5e-5,  # Often higher for cosine

        # ===== EVALUATION & CHECKPOINTING =====
        eval_strategy="steps",
        eval_steps=250,  # More frequent evaluation
        save_strategy="steps",
        save_steps=250,  # More frequent checkpoints
        save_total_limit=5,  # Keep only last 5 checkpoints

        # ===== LOGGING & MONITORING =====
        logging_strategy="steps",
        logging_steps=50,  # More frequent logging
        logging_dir=f"{output_dir}/logs",
        logging_first_step=True,  # Log the very first step

        # ===== MODEL SELECTION =====
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # Alternative if you have accuracy:
        # metric_for_best_model="eval_accuracy",
        # greater_is_better=True,

        # ===== MEMORY & PERFORMANCE OPTIMIZATION =====
        fp16=torch.cuda.is_available(),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),  # For newer GPUs
        tf32=torch.cuda.is_available(),  # Enable tf32 on Ampere GPUs
        dataloader_pin_memory=False,
        dataloader_num_workers=4,  # Parallel data loading
        dataloader_prefetch_factor=2,

        # ===== ADVANCED TRAINING OPTIONS =====
        # Gradient checkpointing to save memory (slower but uses less memory)
        gradient_checkpointing=True,

        # DeepSpeed integration (if using DeepSpeed)
        # deepspeed="ds_config.json",

        # Multi-GPU training
        ddp_find_unused_parameters=False,

        # Early stopping (custom implementation usually needed)
        # You can implement early stopping via callbacks

        # ===== RESUME TRAINING =====
        resume_from_checkpoint=None,  # Path to checkpoint to resume from

        # ===== OTHER CONFIGURATIONS =====
        no_cuda=not torch.cuda.is_available(),
        remove_unused_columns=True,
        label_names=["labels"],  # Explicitly specify label names
        group_by_length=True,  # Group sequences by length for efficiency
        length_column_name="length",  # If you have sequence length column
        prediction_loss_only=False,  # Set to True if only care about loss

        # ===== SEED FOR REPRODUCIBILITY =====
        seed=42,

        # ===== DATA PROCESSING =====
        dataloader_drop_last=True,  # Drop incomplete batches
        past_index=-1,  # Controls how much past to store
    )

    # If you need even more training, consider these additional strategies:
    """
    Additional ways to increase training duration:

    1. Progressive learning rates:
    - Start with 5e-5 for first 5 epochs
    - Reduce to 1e-5 for next 5 epochs  
    - Reduce to 5e-6 for final epochs

    2. Curriculum learning:
    - Start with easy examples
    - Gradually increase difficulty

    3. Multi-stage training:
    - Stage 1: Train on general domain data (3 epochs)
    - Stage 2: Train on specific domain data (5 epochs)
    - Stage 3: Fine-tune on task-specific data (2 epochs)

    4. Use callbacks for custom training control:
    from transformers import TrainerCallback

    class CustomCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            # Custom logic here
            pass

    5. Consider using a learning rate finder first to determine optimal LR
    """

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(
            # Stop after 750 steps (3 × 250) without improvement
            early_stopping_patience=3,
            early_stopping_threshold=0.001
        )]
    )

    # Start training
    print("\n=== Starting Training ===")
    trainer.train()

    # Compute final accuracy on validation set
    print("\n=== Computing Final Accuracy ===")
    eval_results = trainer.evaluate()
    print(f"Validation results: {eval_results}")

    # Optional: Compute train accuracy too
    train_results = trainer.evaluate(train_dataset)
    print(f"Train results: {train_results}")

    # Save the model
    trainer.save_model(f"./model/{LANGUAGE}_character-classifier-quadri-final")
    tokenizer.save_pretrained(
        f"./model/{LANGUAGE}_character-classifier-quadri-final")

    return model, tokenizer


if __name__ == "__main__":
    from time import time
    # Train the model
    for lang in ["lat", "grc", "seals"]:
        print("=========")
        print(f"Training character embedding for language {lang}")
        start_time = time()
        trained_model, trained_tokenizer = train(lang)
        end_time = time()
        print(f"Total required time for training {end_time-start_time}"
              f" language {lang}")
