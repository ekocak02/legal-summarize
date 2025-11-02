# chunker.py

"""
Handles the chunking of long documents for summarization.

This module provides the Chunker class, which can:
1.  Split long documents into smaller, manageable chunks suitable for models
    with limited input length, using a hybrid semantic and sliding-window approach.
2.  Generate "oracle summaries" for each chunk during the training phase by
    extracting the most relevant sentences from the chunk based on the original
    document's summary.
"""

import os
import re
import logging
import nltk
from typing import List, Dict, Any

from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from rouge_score import rouge_scorer

from config import (
    LOGGING_CONFIG,
    DATA_SPLIT_CONFIG,
    CHUNKING_CONFIG,
    STRUCTURAL_SPECIAL_TOKENS
)

# --- Setup ---
logging.basicConfig(
    level=LOGGING_CONFIG['level'],
    format=LOGGING_CONFIG['format'],
    datefmt=LOGGING_CONFIG['datefmt']
)


class Chunker:
    """
    Encapsulates the logic for document chunking and oracle summary creation.
    """
    def __init__(self, config: Dict[str, Any], tokenizer: PreTrainedTokenizerFast = None):
        """
        Initializes the Chunker with a given configuration.

        Args:
            config (Dict[str, Any]): A dictionary containing chunking settings.
        """
        self.config = config
        if tokenizer:
            self.tokenizer = tokenizer
            logging.info("Chunker is using a pre-loaded tokenizer.")
        else:
            logging.info(f"Initializing tokenizer for Chunker: {self.config['model_name']}")
            self.tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
                self.config['model_name']
            )
            logging.info(f"Adding {len(STRUCTURAL_SPECIAL_TOKENS)} special tokens to the Chunker's tokenizer.")
            self.tokenizer.add_special_tokens({
                'additional_special_tokens': STRUCTURAL_SPECIAL_TOKENS
            })
        self.scorer = None

    def _initialize_scorer(self):
        """Initializes the ROUGE scorer on-demand to avoid loading it unnecessarily."""
        if self.scorer is None:
            logging.info("Initializing ROUGE scorer for oracle summary generation.")
            self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def _get_context_header(self, text: str) -> List[int]:
        """
        Creates a context-providing header from the beginning of the document.
        
        This method extracts a fixed number of words to serve as a short title,
        then embeds it in a prompt to give the model context about the document's
        origin.
        Args:
            text (str): The full text of the document.
        
        Returns:
            List[int]: A list of token IDs for the generated context header.
        """
        words = text.split()
        short_title = " ".join(words[:self.config['header_word_count']])

        # Clean up any special tags that might have been included in the title.
        short_title = re.split(r'(\[PARAGRAPH_BREAK\]|\[SECTION_START\])', short_title, 1)[0].strip()

        context_prompt = f"This text is a section from a document titled '{short_title}...'. The section is as follows: "
        header_tokens = self.tokenizer(context_prompt, add_special_tokens=False).input_ids
        
        return header_tokens

    def _chunk_semantically(self, text: str) -> List[str]:
        """
        Splits a long text into chunks using a hybrid semantic approach.

        The process is as follows:
        1.  The text is first split by semantic boundaries (e.g., paragraphs, sections).
        2.  Very short blocks (e.g., titles) are held and prepended to the next block.
        3.  Semantic blocks are accumulated into a buffer. If a block doesn't fit, the buffer is flushed
            to create a chunk.
        4.  If a single semantic block is larger than the chunk size, it's split using a
            traditional sliding window approach.
        5.  Finally, specific rules are applied to merge or drop the last chunk to avoid
            inefficiently small chunks.
            
        Args:
            text (str): The full document text to be chunked.
        
        Returns:
            List[str]: A list of decoded text chunks, ready for the model.
        """
        header_tokens = self._get_context_header(text)
        header_len = len(header_tokens)
        chunk_size = self.config['chunk_size']
        effective_chunk_size = chunk_size - header_len
        
        # A threshold in tokens to identify very short, "dangling" blocks like titles.
        DANGLING_BLOCK_THRESHOLD = 25

        if header_len >= chunk_size:
            logging.warning(f"Context header ({header_len}) is longer than chunk_size ({chunk_size}). Truncating header.")
            header_tokens = header_tokens[:chunk_size - 50]
            header_len = len(header_tokens)
            effective_chunk_size = chunk_size - header_len

        # 1. Split text by semantic separators, keeping the separators.
        separators = r'(\[PARAGRAPH_BREAK\]|\[SECTION_START\]|\[SUBSECTION_[a-z]\])'
        parts = re.split(separators, text, flags=re.IGNORECASE)

        # 2. Reconstruct semantic blocks (text segment + its trailing separator).
        semantic_blocks = []
        i = 0
        while i < len(parts):
            segment = parts[i]
            if i + 1 < len(parts) and parts[i+1].strip(): # Add separator if it exists
                segment += parts[i+1]
            if segment.strip():
                semantic_blocks.append(segment)
            i += 2
        
        if not semantic_blocks: # If no splits, treat the whole text as one block.
            semantic_blocks = [text]

        # 3. Process blocks to form chunks.
        all_chunks_tokens = []
        buffer_tokens = []
        dangling_tokens = [] # To hold very short blocks.

        for block in semantic_blocks:
            block_tokens = self.tokenizer(block, add_special_tokens=False).input_ids

            if dangling_tokens:
                block_tokens = dangling_tokens + block_tokens
                dangling_tokens = []

            if len(block_tokens) < DANGLING_BLOCK_THRESHOLD:
                dangling_tokens.extend(block_tokens)
                continue

            # If the block is oversized, flush buffer and split the block itself.
            if len(block_tokens) > effective_chunk_size:
                if buffer_tokens:
                    all_chunks_tokens.append(header_tokens + buffer_tokens)
                    buffer_tokens = []
                
                step = chunk_size - self.config['overlap']
                for i in range(0, len(block_tokens), step):
                    chunk_content = block_tokens[i : i + effective_chunk_size]
                    all_chunks_tokens.append(header_tokens + chunk_content)
            # If block fits, add it to the buffer.
            else:
                if len(buffer_tokens) + len(block_tokens) > effective_chunk_size:
                    if buffer_tokens:
                        all_chunks_tokens.append(header_tokens + buffer_tokens)
                    buffer_tokens = block_tokens
                else:
                    buffer_tokens.extend(block_tokens)
        
        # Flush any remaining tokens in both buffers.
        if dangling_tokens:
            buffer_tokens.extend(dangling_tokens)
        if buffer_tokens:
            all_chunks_tokens.append(header_tokens + buffer_tokens)

        if not all_chunks_tokens:
            return []

        # 4. Apply special handling for the final chunk.
        final_chunks_tokens = all_chunks_tokens
        if len(final_chunks_tokens) > 1:
            if (len(final_chunks_tokens[-1]) + len(final_chunks_tokens[-2])) < self.config.get('merge_threshold', 1024):
                merged_content = final_chunks_tokens[-1][header_len:]
                final_chunks_tokens[-2].extend(merged_content)
                final_chunks_tokens.pop()
                logging.debug("Merged the last two chunks.")
        
        min_content_tokens = self.config.get('min_chunk_tokens', 50)
        if len(final_chunks_tokens) > 0 and len(final_chunks_tokens[-1]) < (header_len + min_content_tokens):
            final_chunks_tokens.pop()
            logging.info("Popped the last chunk as it was too short.")
            
        return self.tokenizer.batch_decode(final_chunks_tokens, skip_special_tokens=True)

    def _create_oracle_summary(self, chunk_text: str, original_summary: str) -> str:
        """
        Generates an oracle summary for a text chunk.
        
        It scores each sentence in the chunk against the original document's summary
        using ROUGE-L, selects the top N sentences, and reorders them to match their
        original sequence in the chunk.
        """
        self._initialize_scorer()

        chunk_sentences = nltk.sent_tokenize(chunk_text)
        if not chunk_sentences:
            return ""

        scores = [
            self.scorer.score(original_summary, sent)['rougeL'].fmeasure
            for sent in chunk_sentences
        ]
        
        scored_sentences = sorted(zip(scores, chunk_sentences), key=lambda x: x[0], reverse=True)
        
        top_sentences = [sent for _, sent in scored_sentences[:self.config['oracle_sentence_count']]]
        
        oracle_summary = " ".join(sorted(top_sentences, key=lambda s: chunk_text.find(s)))
        return oracle_summary

    # --- Public Methods ---

    def chunk_for_inference(self, processed_text: str) -> List[str]:
        """
        Public method to chunk a text for inference. No summary is required.
        """
        return self._chunk_semantically(processed_text)

    def chunk_for_training(self, example: Dict) -> Dict:
        """
        Public method to process a single example for training.
        
        It chunks the text and creates an oracle summary for each chunk.
        Designed for use with Hugging Face Datasets' .map() method.
        """
        document_text = example.get('text', '')
        summary_text = example.get('summary', '')
        
        if not document_text or not summary_text:
            return {'processed_chunks': []}
            
        text_chunks = self._chunk_semantically(document_text)
        new_pairs = []
        for chunk in text_chunks:
            oracle_summary = self._create_oracle_summary(chunk, summary_text)
            if oracle_summary:
                new_pairs.append({'text': chunk, 'summary': oracle_summary})
        
        return {'processed_chunks': new_pairs}

def main():
    """Main function to run the chunking pipeline as a script."""
    logging.info("--- Starting Chunking Pipeline ---")
    
    chunker = Chunker(config=CHUNKING_CONFIG)
    
    logging.info("Loading datasets from split directory...")
    split_dir = DATA_SPLIT_CONFIG['split_data_dir']
    data_files = {
        split: os.path.join(split_dir, f"{split}.jsonl")
        for split in ['train', 'validation']
    }
    raw_datasets = load_dataset('json', data_files=data_files)
    
    final_splits = {}
    for split_name, dataset in raw_datasets.items():
        logging.info(f"--- Processing split: {split_name} ---")
        
        dataset_with_length = dataset.map(
            lambda ex: {'token_length': len(chunker.tokenizer(ex["text"], add_special_tokens=False).input_ids)},
            num_proc=os.cpu_count()
        )
        
        long_texts = dataset_with_length.filter(
            lambda ex: ex['token_length'] > chunker.config['chunk_size']
        )
        short_texts = dataset_with_length.filter(
            lambda ex: ex['token_length'] <= chunker.config['chunk_size']
        )
        logging.info(f"Found {len(long_texts)} long texts and {len(short_texts)} short texts.")
        
        if len(long_texts) > 0:
            processed_long_texts = long_texts.map(
                chunker.chunk_for_training,
                remove_columns=long_texts.column_names,
                num_proc=os.cpu_count()
            )
            
            processed_chunks_list = [
                chunk for example in processed_long_texts
                for chunk in example['processed_chunks']
            ]
            flat_long_texts_split = Dataset.from_list(processed_chunks_list)
        else:
            flat_long_texts_split = Dataset.from_dict({'text': [], 'summary': []})
            
        short_texts_cleaned = short_texts.remove_columns(['token_length'])
        final_dataset = concatenate_datasets([short_texts_cleaned, flat_long_texts_split])
        final_splits[split_name] = final_dataset
        
        logging.info(f"Finished processing '{split_name}'. Final sample count: {len(final_dataset)}")

    output_dir = DATA_SPLIT_CONFIG['split_data_dir']
    os.makedirs(output_dir, exist_ok=True)
    for split_name, dataset_split in final_splits.items():
        output_path = os.path.join(output_dir, f"{split_name}_chunked.jsonl")
        dataset_split.to_json(output_path, orient='records', lines=True)
        logging.info(f"Saved final chunked '{split_name}' set to '{output_path}'")
        
    logging.info("--- Chunking Pipeline Finished Successfully ---")


if __name__ == "__main__":
    main()