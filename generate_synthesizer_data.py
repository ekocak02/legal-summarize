# generate_synthesizer_data.py
"""
This script prepares data for training the Flan-T5 Synthesizer model.

It performs steps 3 and 4 of the plan:
1.  Loads the best-trained BART model (Chunk Summarizer).
2.  Reads the original *training* dataset ('train.jsonl').
3.  Applies the hierarchical summarization process (clean, chunk,
    summarize chunks, combine) to each document.
4.  Saves these combined BART summaries as the 'text' column for
    T5 to learn from.
5.  Saves the original reference summary as the 'summary' column.
6.  Writes the results to the .jsonl file to be used in T5 training.
"""

import os
import logging
import json
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Dict, Any

# Import project modules and configurations
from config import (
    EVALUATION_CONFIG,       # To get the BART model path and generation parameters
    DATA_SPLIT_CONFIG,       # To get the source 'train.jsonl' path
    SYNTHESIZER_TRAINING_CONFIG, # To get the output file path and T5 model name
    LOGGING_CONFIG,
    DATA_PROCESSING_CONFIG,
    FILTERING_CONFIG,
    CHUNKING_CONFIG
)
from processor import DataProcessor
from chunker import Chunker

# --- Setup ---
logging.basicConfig(level=LOGGING_CONFIG['level'], format=LOGGING_CONFIG['format'], datefmt=LOGGING_CONFIG['datefmt'])

def _generate_summaries_batch(
    texts: List[str], model: Any, tokenizer: Any,
    gen_params: Dict[str, Any], device: str
) -> List[str]:
    """
    Helper function to generate summaries for a batch of texts.
    """
    inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=1024
    ).to(device)

    with torch.no_grad(): # Disable gradient calculation in inference mode
        summary_ids = model.generate(**inputs, **gen_params)
        
    return tokenizer.batch_decode(summary_ids, skip_special_tokens=True)

def main():
    """Main data generation script."""
    logging.info("--- T5 Synthesizer Data Generation Script Started (Kaggle Filtered Mode) ---")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # 1. Load models and tools
    bart_model_path = EVALUATION_CONFIG['chunk_summarizer_model_path']
    logging.info(f"Loading BART model (Chunk Summarizer): {bart_model_path}")
    
    bart_tokenizer = AutoTokenizer.from_pretrained(bart_model_path)
    bart_model = AutoModelForSeq2SeqLM.from_pretrained(bart_model_path).to(device).eval()

    # Load T5's own tokenizer for filtering
    t5_model_name = SYNTHESIZER_TRAINING_CONFIG['base_model']
    logging.info(f"Loading base T5 tokenizer (for filtering): {t5_model_name}")
    synthesis_tokenizer = AutoTokenizer.from_pretrained(t5_model_name)

    # Initialize DataProcessor and Chunker with BART tokenizer
    processor = DataProcessor(DATA_PROCESSING_CONFIG, FILTERING_CONFIG, tokenizer=bart_tokenizer)
    chunker = Chunker(CHUNKING_CONFIG, tokenizer=bart_tokenizer)

    # 2. Load source training data
    # We are generating summaries from 'train.jsonl' to train T5
    source_data_path = os.path.join(DATA_SPLIT_CONFIG['split_data_dir'], 'train.jsonl')
    logging.info(f"Loading source training data: {source_data_path}")
    dataset = load_dataset("json", data_files=source_data_path)['train']

    # 3. Prepare the output file
    output_path = SYNTHESIZER_TRAINING_CONFIG['train_data_path']
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    logging.info(f"Generated summaries will be written to: {output_path}")

    # 4. Loop over the dataset and generate summaries
    batch_size = 16 
    total_processed = 0
    total_kept = 0

    with open(output_path, 'w', encoding='utf-8') as f_out:
        for i in tqdm(range(0, len(dataset), batch_size), desc="Generating hierarchical summaries"):
            batch = dataset[i : i + batch_size]
            
            all_reference_summaries = batch['summary']
            texts_to_chunk = batch['text']
            
            chunks_for_bart_model = []
            chunk_group_indices = [] # Tracks which chunks belong to which document

            for doc_text in texts_to_chunk:
                total_processed += 1
                
                if not doc_text:
                    chunk_group_indices.append(0) # 0 chunks for this document
                    continue

                cleaned_text = processor.process_text(doc_text, title=None)
                chunks = chunker.chunk_for_inference(cleaned_text)

                if not chunks:
                    chunk_group_indices.append(0)
                    continue
                
                prompt_for_bart = "Summarize the following section of a legal document: "
                chunks_with_prompt = [prompt_for_bart + chunk for chunk in chunks]

                chunks_for_bart_model.extend(chunks_with_prompt)
                chunk_group_indices.append(len(chunks_with_prompt))

            if not chunks_for_bart_model:
                continue

            # Step B: Summarize chunks with BART (in batch)
            bart_summaries = _generate_summaries_batch(
                chunks_for_bart_model, 
                bart_model, 
                bart_tokenizer,
                EVALUATION_CONFIG['chunk_generation_params'],
                device
            )
            
            # Step C: Combine, Filter, and Save
            bart_summary_index = 0
            for doc_index, num_chunks in enumerate(chunk_group_indices):
                if num_chunks == 0:
                    continue
                
                # Get the intermediate summaries for this document
                doc_bart_summaries = bart_summaries[bart_summary_index : bart_summary_index + num_chunks]
                bart_summary_index += num_chunks
                
                combined_summaries = " ".join(doc_bart_summaries)

                # T5 token limit filter
                synthesizer_input_ids = synthesis_tokenizer(
                    combined_summaries, add_special_tokens=False
                ).input_ids
                token_length = len(synthesizer_input_ids)

                # If it exceeds T5's max input limit (1024), don't include this data in training
                if token_length > SYNTHESIZER_TRAINING_CONFIG['max_input_length']:
                    logging.debug(f"Skipping document {i+doc_index}. (Token: {token_length} > {SYNTHESIZER_TRAINING_CONFIG['max_input_length']})")
                    continue

                # Step D: Save as training data for T5
                result_for_t5 = {
                    "text": combined_summaries,
                    "summary": all_reference_summaries[doc_index]
                }
                f_out.write(json.dumps(result_for_t5, ensure_ascii=False) + '\n')
                total_kept += 1

    
    logging.info(f"--- Filtered Data Generation Complete. File saved: {output_path} ---")
    logging.info(f"Total documents processed: {total_processed}")
    logging.info(f"Documents passed filter (saved): {total_kept}")

if __name__ == "__main__":
    main()