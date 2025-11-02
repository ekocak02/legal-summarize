# evaluate_scores.py
"""
This script calculates ROUGE and BERTScore for model predictions.

It takes a JSONL file as input, which should contain 'generated_summary'
and 'reference_summary' columns.

Prerequisites:
pip install datasets evaluate rouge-score bert-score transformers torch nltk
"""

import os
import argparse
import logging
import nltk
import torch
import pandas as pd
from datasets import load_dataset
from evaluate import load as load_metric
from bert_score import score as bert_score

from config import LOGGING_CONFIG

# --- Setup ---
logging.basicConfig(level=LOGGING_CONFIG['level'], format=LOGGING_CONFIG['format'], datefmt=LOGGING_CONFIG['datefmt'])

# Ensure 'punkt' is available for ROUGE calculation
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    logging.info("NLTK 'punkt' model not found. Downloading...")
    nltk.download('punkt', quiet=True)

def calculate_rouge_scores(predictions: list[str], references: list[str]) -> dict:
    """Calculates ROUGE scores."""
    logging.info("Calculating ROUGE scores...")
    rouge_metric = load_metric("rouge")

    # NLTK sentence tokenization for ROUGE
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in predictions]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in references]

    result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {key: value * 100 for key, value in result.items()}
    result = {k: round(v, 4) for k, v in result.items()}
    logging.info("ROUGE calculation finished.")
    return result

def calculate_bert_scores(predictions: list[str], references: list[str], lang: str = "en") -> dict:
    """Calculates BERTScore."""
    logging.info(f"Calculating BERTScore for language: {lang}...")
    
    # Use GPU if available
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # For Turkish, you might use 'bert-base-turkish-cased'
    # For English, a common default is 'roberta-large'
    model_type = "dbmdz/bert-base-turkish-cased" if lang == "tr" else "roberta-large"

    P, R, F1 = bert_score(
        predictions,
        references,
        model_type=model_type,
        lang=lang,
        verbose=True,
        device=device
    )
    
    result = {
        "bert_precision": round(P.mean().item() * 100, 4),
        "bert_recall": round(R.mean().item() * 100, 4),
        "bert_f1": round(F1.mean().item() * 100, 4),
    }
    logging.info("BERTScore calculation finished.")
    return result

def main(args):
    """Main function to load data and compute scores."""
    if not os.path.exists(args.prediction_file):
        logging.error(f"Prediction file not found at: {args.prediction_file}")
        return

    logging.info(f"Loading predictions from: {args.prediction_file}")
    dataset = load_dataset("json", data_files=args.prediction_file)["train"]

    # Extract columns
    predictions = list(dataset[args.prediction_col])
    references = list(dataset[args.reference_col])

    logging.info(f"Loaded {len(predictions)} samples for evaluation.")

    # --- Calculate Scores ---
    all_scores = {}
    
    if "rouge" in args.metrics:
        rouge_results = calculate_rouge_scores(predictions, references)
        all_scores.update(rouge_results)
        
    if "bert" in args.metrics:
        bert_results = calculate_bert_scores(predictions, references, lang=args.lang)
        all_scores.update(bert_results)

    # --- Display Results ---
    logging.info("\n--- Overall Evaluation Scores ---")
    results_df = pd.DataFrame([all_scores])
    print(results_df.to_string(index=False))
    logging.info("---------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate summarization model predictions.")
    parser.add_argument(
        "prediction_file",
        type=str,
        help="Path to the JSONL file containing predictions and references.",
    )
    parser.add_argument(
        "--prediction_col",
        type=str,
        default="generated_summary",
        help="Name of the column with generated summaries.",
    )
    parser.add_argument(
        "--reference_col",
        type=str,
        default="reference_summary",
        help="Name of the column with reference summaries.",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        help="Language of the text for BERTScore (e.g., 'en', 'tr').",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["rouge", "bert"],
        choices=["rouge", "bert"],
        help="List of metrics to compute.",
    )
    
    args = parser.parse_args()
    main(args)