# evaluation.py
"""
This script runs the end-to-end hierarchical summarization pipeline on a test dataset.

It loads the trained BART (chunk summarizer) and Flan-T5 (synthesizer) models
to perform full inference. It processes and chunks the test data, generates
intermediate summaries with BART, and then uses Flan-T5 to synthesize
these into a final summary.

The output file saves the source text, reference summary, intermediate
BART summary, and the final generated summary for evaluation and debugging.
"""

import os
import logging
import json
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Import project-specific modules
from config import (
    EVALUATION_CONFIG,
    LOGGING_CONFIG,
    DATA_PROCESSING_CONFIG,
    FILTERING_CONFIG,
    CHUNKING_CONFIG
)
from processor import DataProcessor
from chunker import Chunker

# --- Setup ---
logging.basicConfig(level=LOGGING_CONFIG['level'], format=LOGGING_CONFIG['format'], datefmt=LOGGING_CONFIG['datefmt'])

class SummarizationEvaluator:
    """Encapsulates the entire evaluation pipeline for the hierarchical summarization model."""

    def __init__(self, config: dict):
        """
        Initializes the evaluator by loading models, tokenizers, and processors.
        
        Args:
            config (dict): A dictionary containing evaluation settings, typically from config.py.
        """
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {self.device}")

        self._load_models_and_tokenizers()
        
        # Initialize data processing tools by PASSING the already-loaded tokenizer
        self.processor = DataProcessor(
            DATA_PROCESSING_CONFIG,
            FILTERING_CONFIG,
            tokenizer=self.chunk_summarizer_tokenizer
        )
        self.chunker = Chunker(
            CHUNKING_CONFIG,
            tokenizer=self.chunk_summarizer_tokenizer
        )

    def _load_models_and_tokenizers(self):
        """Loads the chunk summarizer and synthesizer models and their tokenizers."""
        logging.info(f"Loading chunk summarizer from: {self.config['chunk_summarizer_model_path']}")
        self.chunk_summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config['chunk_summarizer_model_path']
        ).to(self.device)
        self.chunk_summarizer_tokenizer = AutoTokenizer.from_pretrained(
            self.config['chunk_summarizer_model_path']
        )

        logging.info(f"Loading synthesizer model from: {self.config['synthesis_model_path']}")
        self.synthesis_model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config['synthesis_model_path']
        ).to(self.device)
        self.synthesis_tokenizer = AutoTokenizer.from_pretrained(
            self.config['synthesis_model_path']
        )

    def _summarize_text_batch(self, texts: list[str], model, tokenizer, gen_params: dict) -> list[str]:
        """Generates summaries for a batch of texts."""
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(self.device)

        with torch.no_grad():
            summary_ids = model.generate(**inputs, **gen_params)
        
        return tokenizer.batch_decode(summary_ids, skip_special_tokens=True)

    def _run_synthesis_step(self, text_to_synthesize: str, prompt_template: str, gen_params: dict) -> str:
        """Applies a single synthesis step using the Flan-T5 model."""
        prompt = prompt_template.format(chunk_summaries=text_to_synthesize)
        return self._summarize_text_batch([prompt], self.synthesis_model, self.synthesis_tokenizer, gen_params)[0]
        
    def run_evaluation(self):
        """Executes the full evaluation pipeline."""
        logging.info("Starting evaluation pipeline...")
        
        # 1. Load dataset
        dataset = load_dataset("json", data_files=self.config['test_data_path'])['train'].select(range(100))
        
        # 2. Optional: Filter the dataset based on token counts, mimicking the training setup
        if self.config.get('apply_filtering_to_test_set', False):
            logging.info("Applying filtering to the test set as configured.")
            
            # We need a tokenizer to count tokens for filtering. We can use the chunk summarizer's.
            tokenizer_for_filtering = self.chunk_summarizer_tokenizer
            
            # Add token counts to the dataset
            def add_token_counts(example):
                example['text_token_count'] = len(tokenizer_for_filtering.encode(example['text']))
                example['summary_token_count'] = len(tokenizer_for_filtering.encode(example['summary']))
                return example

            dataset_with_counts = dataset.map(add_token_counts, num_proc=os.cpu_count())

            # Define filter limits from the global FILTERING_CONFIG
            min_text = FILTERING_CONFIG['min_text_tokens']
            max_text = FILTERING_CONFIG['max_text_tokens']
            min_summary = FILTERING_CONFIG['min_summary_tokens']
            max_summary = FILTERING_CONFIG['max_summary_tokens']

            # Apply the filter
            original_size = len(dataset_with_counts)
            dataset = dataset_with_counts.filter(
                lambda ex: (min_text <= ex['text_token_count'] <= max_text) and \
                           (min_summary <= ex['summary_token_count'] <= max_summary),
                num_proc=os.cpu_count()
            )
            logging.info(f"Filtered test set from {original_size} to {len(dataset)} samples.")
        
        # 3. Optional: Subsample for quick tests
        if self.config.get('max_test_samples'):
            dataset = dataset.select(range(self.config['max_test_samples']))
        
        logging.info(f"Proceeding with {len(dataset)} samples for evaluation.")

        # 2. Prepare output file
        # GÜNCELLENDİ: 'config.py' dosyasındaki doğru anahtarı oku
        output_path = self.config['flan_t5_predictions_output_path']
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        # 3. Process each example and generate summary
        with open(output_path, 'w', encoding='utf-8') as f_out:
            for example in tqdm(dataset, desc="Evaluating Samples"):
                source_text = example.get('text', '')
                reference_summary = example.get('summary', '')

                if not source_text or not reference_summary:
                    continue

                # Step A: Clean and chunk the source text
                cleaned_text = self.processor.process_text(source_text, example.get('title'))
                chunks = self.chunker.chunk_for_inference(cleaned_text)

                prompt_for_bart = "Summarize the following section of a legal document: "
                chunks_with_prompt = [prompt_for_bart + chunk for chunk in chunks]

                # Step B: Summarize each chunk with BART
                chunk_summaries = self._summarize_text_batch(
                    chunks_with_prompt, 
                    self.chunk_summarizer_model, 
                    self.chunk_summarizer_tokenizer,
                    self.config['chunk_generation_params']
                )
                # This is the intermediate output from the BART model
                combined_summaries = " ".join(chunk_summaries)

                # Step C: Conditional, multi-layer synthesis with Flan-T5
                synthesizer_input_ids = self.synthesis_tokenizer(combined_summaries, return_tensors="pt").input_ids
                
                if synthesizer_input_ids.shape[1] > self.config['synthesizer_input_max_tokens']:
                    # Intermediate synthesis needed
                    logging.debug(f"Input too long ({synthesizer_input_ids.shape[1]} tokens). Applying intermediate synthesis.")
                    
                    summary_chunks = self.chunker.chunk_for_inference(combined_summaries)
                    
                    intermediate_summaries = []
                    for summary_chunk in summary_chunks:
                        intermediate_summary = self._run_synthesis_step(
                            summary_chunk,
                            self.config['intermediate_synthesis_prompt'],
                            self.config['intermediate_synthesis_generation_params']
                        )
                        intermediate_summaries.append(intermediate_summary)
                    
                    final_input_text = " ".join(intermediate_summaries)
                else:
                    final_input_text = combined_summaries

                # Final synthesis step
                # This is the final output from the T5 model
                generated_summary = self._run_synthesis_step(
                    final_input_text,
                    self.config['final_synthesis_prompt'],
                    self.config['final_synthesis_generation_params']
                )

                # 4. Save results
                # Save both BART (intermediate) and T5 (final) summaries for analysis
                result = {
                    "source_text": source_text,
                    "reference_summary": reference_summary,
                    "bart_intermediate_summary": combined_summaries, # Added for debugging/analysis
                    "generated_summary": generated_summary
                }
                f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        logging.info(f"Evaluation finished. Predictions saved to '{output_path}'")

def main():
    """Initializes and runs the evaluator."""
    evaluator = SummarizationEvaluator(config=EVALUATION_CONFIG)
    evaluator.run_evaluation()

if __name__ == "__main__":
    main()