# train_flant5.py
"""
This script fine-tunes a Flan-T5 model for the synthesis task.

It now uses a hybrid dataset:
1.  Original, naturally short documents.
2.  BART-generated and combined chunk summaries.
"""

import os
import logging
import nltk
import numpy as np
import evaluate
import torch
from datasets import load_dataset, concatenate_datasets # concatenate_datasets added
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback
)
from typing import Dict, Any

from config import SYNTHESIZER_TRAINING_CONFIG, LOGGING_CONFIG, STRUCTURAL_SPECIAL_TOKENS

# --- Setup ---
logging.basicConfig(level=LOGGING_CONFIG['level'], format=LOGGING_CONFIG['format'], datefmt=LOGGING_CONFIG['datefmt'])

class SynthesizerTrainer:
    """A class to encapsulate the Flan-T5 synthesizer model training pipeline."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['base_model'])
        self.rouge_metric = evaluate.load("rouge")
        logging.info("Initialized SynthesizerTrainer for Flan-T5.")

    def _preprocess_data(self, examples):
        """Tokenize texts and summaries, adding the task-specific prefix for T5."""
        # Add the task-specific prefix required by T5 models.
        prefix = self.config.get('task_prefix', '') # Safely get prefix
        if not prefix:
            logging.warning("Task prefix is not set in the config. T5 models generally perform better with one.")
        
        inputs = [prefix + doc for doc in examples["text"]]
        
        model_inputs = self.tokenizer(
            inputs,
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
        
        # Replace -100 (ignore token) with pad token for decoding
        preds = np.where(preds != -100, preds, self.tokenizer.pad_token_id)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # NLTK sentence tokenization for ROUGE
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
        
        # Compute ROUGE scores
        result = self.rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {key: value * 100 for key, value in result.items()}
        
        # Also compute the average generation length
        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        
        return {k: round(v, 4) for k, v in result.items()}

    def train(self):
        """Executes the full training and evaluation pipeline for the synthesizer."""
        # 1. Load Tokenizer and add special tokens
        logging.info(f"Loading tokenizer from '{self.config['base_model']}'")
        logging.info(f"Adding {len(STRUCTURAL_SPECIAL_TOKENS)} new special tokens to the tokenizer.")
        self.tokenizer.add_special_tokens({
            'additional_special_tokens': STRUCTURAL_SPECIAL_TOKENS
        })

        logging.info(f"Loading base model: '{self.config['base_model']}'")
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config['base_model'])

        logging.info("Resizing model token embeddings to match the new tokenizer size.")
        model.resize_token_embeddings(len(self.tokenizer))

        # --- Hybrid Data Loading Logic ---
        logging.info("Loading hybrid dataset for Flan-T5 training...")

        # 1. Load original short documents (For both train and validation)
        logging.info("Loading Part 1: Original short documents...")
        
        # CORRECTED: Changed 'split_data_dir' to 'processed_data_dir' to match config.py
        data_dir = self.config['processed_data_dir']

        if not data_dir or not os.path.exists(data_dir):
            logging.error(f"Data directory '{data_dir}' not found. Please set 'processed_data_dir' in config.py.")
            return

        data_files = {
            'train': os.path.join(data_dir, 'train.jsonl'),
            'validation': os.path.join(data_dir, 'validation.jsonl'),
        }
        original_dataset = load_dataset('json', data_files=data_files)
        
        logging.info(f"Filtering original dataset to include only documents with <= {self.config['data_filter_max_tokens']} tokens.")
        
        # Calculate token length
        original_dataset_with_length = original_dataset.map(
            lambda ex: {'token_length': len(self.tokenizer(ex["text"]).input_ids)},
            num_proc=os.cpu_count()
        )
        # Apply the filter
        short_docs_dataset = original_dataset_with_length.filter(
            lambda ex: ex['token_length'] <= self.config['data_filter_max_tokens'],
            num_proc=os.cpu_count()
        )
        
        # The validation set will CONSIST ONLY of short documents
        final_validation_dataset = short_docs_dataset['validation']
        logging.info(f"Validation set (short docs only): {len(final_validation_dataset)} samples.")
        
        # 2. Load BART-generated summaries (For train only)
        logging.info("Loading Part 2: BART-generated summaries...")
        generated_data_path = self.config['train_data_path']
        if not os.path.exists(generated_data_path):
            logging.error(f"BART-generated summary file not found at '{generated_data_path}'.")
            logging.error("Please run 'generate_synthesizer_data.py' first.")
            return
            
        generated_dataset = load_dataset('json', data_files={'train': generated_data_path})['train']
        logging.info(f"Loaded {len(generated_dataset)} BART-generated summaries.")

        # 3. Concatenate the training datasets
        # Original short training documents + BART-generated summaries
        final_train_dataset = concatenate_datasets([
            short_docs_dataset['train'],
            generated_dataset
        ])
        
        logging.info(f"Hybrid training set created. Total samples: {len(final_train_dataset)}")
        logging.info(f"Final training set size: {len(final_train_dataset)}")
        logging.info(f"Final validation set size: {len(final_validation_dataset)}")

        # Collect the combined datasets in a DatasetDict
        hybrid_dataset = {
            "train": final_train_dataset,
            "validation": final_validation_dataset
        }

        logging.info("Tokenizing the hybrid dataset...")
        tokenized_dataset = {}
        tokenized_dataset["train"] = hybrid_dataset["train"].map(
            self._preprocess_data, batched=True, num_proc=os.cpu_count(), remove_columns=hybrid_dataset["train"].column_names
        )
        tokenized_dataset["validation"] = hybrid_dataset["validation"].map(
            self._preprocess_data, batched=True, num_proc=os.cpu_count(), remove_columns=hybrid_dataset["validation"].column_names
        )

        # Apply subsampling for quick debugging runs if specified
        train_dataset = tokenized_dataset['train']
        eval_dataset = tokenized_dataset['validation']
        
        if self.config.get('max_train_samples'):
            train_dataset = train_dataset.select(range(self.config['max_train_samples']))
        if self.config.get('max_eval_samples'):
            eval_dataset = eval_dataset.select(range(self.config['max_eval_samples']))

        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=model)
        
        training_args_dict = self.config['training_arguments']
        training_args_dict['fp16'] = torch.cuda.is_available() # Automatically enable fp16 if GPU is available
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

        logging.info("Starting Flan-T5 synthesizer model training...")
        trainer.train()
        logging.info("Training finished.")

        logging.info(f"Saving the best model to '{self.config['final_model_dir']}'")
        trainer.save_model(self.config['final_model_dir'])
        self.tokenizer.save_pretrained(self.config['final_model_dir'])
        logging.info("Synthesizer model saved successfully.")

def main():
    """Initializes and runs the synthesizer trainer."""
    trainer = SynthesizerTrainer(config=SYNTHESIZER_TRAINING_CONFIG)
    trainer.train()

if __name__ == "__main__":
    main()