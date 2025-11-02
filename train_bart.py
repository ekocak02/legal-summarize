# train_bart.py
"""
This script handles the training of the sequence-to-sequence summarization model.
It encapsulates the entire training pipeline, from data loading and preprocessing
to evaluation and model saving, into a configurable ModelTrainer class.
"""

import os
import logging
import nltk
import evaluate
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
    PreTrainedTokenizer
)
from typing import Dict, Any

from config import TRAINING_CONFIG, LOGGING_CONFIG, STRUCTURAL_SPECIAL_TOKENS

# --- Setup ---
logging.basicConfig(
    level=LOGGING_CONFIG['level'],
    format=LOGGING_CONFIG['format'],
    datefmt=LOGGING_CONFIG['datefmt']
)

# Ensure 'punkt' is available for metric calculation
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    logging.info("NLTK 'punkt' model not found. Downloading...")
    nltk.download('punkt')

class ModelTrainer:
    """
    A class to encapsulate the model training pipeline.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the ModelTrainer with configurations.
        """
        self.config = config
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(self.config['base_model'])
        self.rouge_metric = evaluate.load("rouge")

    def _preprocess_data(self, examples):
        """Tokenize the text and summary fields."""
        model_inputs = self.tokenizer(
            examples["text"],
            max_length=self.config['max_input_length'],
            truncation=True
        )
        labels = self.tokenizer(
            text_target=examples["summary"],
            max_length=self.config['max_target_length'],
            truncation=True
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def _compute_metrics(self, eval_preds):
        """Computes ROUGE scores for evaluation."""
        preds, labels = eval_preds
        preds = np.where(preds != -100, preds, self.tokenizer.pad_token_id)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Prepare sentences for ROUGE scoring
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

        result = self.rouge_metric.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )
        result = {key: value * 100 for key, value in result.items()}

        # Add mean generated length to metrics
        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}

    def train(self):
        """
        Executes the full training and evaluation pipeline.
        """
        # 1. Load Tokenizer and add special tokens
        logging.info(f"Loading tokenizer from '{self.config['base_model']}'")
        logging.info(f"Adding {len(STRUCTURAL_SPECIAL_TOKENS)} new special tokens to the tokenizer.")
        self.tokenizer.add_special_tokens({
            'additional_special_tokens': STRUCTURAL_SPECIAL_TOKENS
        })

        # 2. Load Model
        logging.info(f"Loading base model from '{self.config['base_model']}'")
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config['base_model'])

        # 3. Resize model embeddings
        logging.info("Resizing model token embeddings to match the new tokenizer size.")
        model.resize_token_embeddings(len(self.tokenizer))

        # 4. Load and Preprocess Data
        logging.info("Loading and preprocessing chunked dataset...")
        data_dir = self.config['chunked_data_dir']
        data_files = {
            'train': os.path.join(data_dir, 'train_chunked.jsonl'),
            'validation': os.path.join(data_dir, 'validation_chunked.jsonl'),
        }
        dataset = load_dataset('json', data_files=data_files)
        
        tokenized_dataset = dataset.map(self._preprocess_data, batched=True, num_proc=os.cpu_count())

        # Subsample if configured (for quick tests)
        # Uses 'max_train_samples' and 'max_eval_samples' from config.py.
        # If set to 'None', the entire dataset is used.
        train_dataset = tokenized_dataset['train']
        eval_dataset = tokenized_dataset['validation']
        
        if self.config.get('max_train_samples'):
            train_dataset = train_dataset.select(range(self.config['max_train_samples']))
        if self.config.get('max_eval_samples'):
            eval_dataset = eval_dataset.select(range(self.config['max_eval_samples']))

        # 5. Set up Trainer
        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=model)
        
        # Unpack training arguments directly from config
        training_args_dict = self.config['training_arguments']
        training_args_dict['fp16'] = torch.cuda.is_available() # Dynamically set fp16

        training_args = Seq2SeqTrainingArguments(**training_args_dict)

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        # 6. Start Training
        logging.info("Starting model training...")
        trainer.train()
        logging.info("Training finished.")

        # 7. Save the best model
        logging.info(f"Saving the best model to '{self.config['final_model_dir']}'")
        trainer.save_model(self.config['final_model_dir'])
        self.tokenizer.save_pretrained(self.config['final_model_dir'])
        logging.info("Model saved successfully.")


def main():
    """Main function to initialize and run the trainer."""
    trainer = ModelTrainer(config=TRAINING_CONFIG)
    trainer.train()


if __name__ == "__main__":
    main()
