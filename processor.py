# processor.py

"""
Contains the DataProcessor class responsible for cleaning, transforming,
and preparing the legal text data for model training and inference.
"""

import os
import re
import logging
from typing import List, Dict, Any

import ftfy
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerFast

# Import configurations from the central config file
from config import (
    DATA_PROCESSING_CONFIG,
    FILTERING_CONFIG,
    LOGGING_CONFIG,
    TEXT_COLUMN,
    SUMMARY_COLUMN,
    TITLE_COLUMN,
)

# Setup logger
logging.basicConfig(
    level=LOGGING_CONFIG['level'],
    format=LOGGING_CONFIG['format'],
    datefmt=LOGGING_CONFIG['datefmt']
)


class DataProcessor:
    """
    A class to encapsulate all data processing logic.

    This class handles text cleaning, structural normalization, token counting,
    and dataset filtering. It is designed to be reusable for training,
    evaluation, and inference pipelines.
    """

    # --- Regex patterns are defined as class attributes ---
    # They are part of the cleaning logic, not external configuration.
    _REGEX_PATTERNS_TO_REMOVE = [
        # Standard preamble in many US bills, contains no summary information.
        re.compile(
            r'be it enacted by the senate and house of representatives of the united states of america in congress assembled,',
            flags=re.IGNORECASE | re.MULTILINE
        ),
        # Common page number markers found in scraped text.
        re.compile(r'\[page\s+\w\d+\]', flags=re.IGNORECASE),
        # Line numbers at the beginning of a line (e.g., "1 Resolved, That...").
        re.compile(r'^\s*\d+\s+', flags=re.MULTILINE),
    ]

    def __init__(self, data_config: Dict[str, Any], filter_config: Dict[str, Any], tokenizer: PreTrainedTokenizerFast = None):
        """
        Initializes the DataProcessor.

        Args:
            data_config (Dict[str, Any]): Configuration for data processing steps.
            filter_config (Dict[str, Any]): Configuration for filtering and tokenization.
            tokenizer (PreTrainedTokenizerFast, optional): An already initialized tokenizer. 
                                                          If None, one will be loaded. Defaults to None.
        """
        self.data_config = data_config
        self.filter_config = filter_config
        
        if tokenizer:
            self.tokenizer = tokenizer
            logging.info("DataProcessor is using a pre-loaded tokenizer.")
        else:
            logging.info(f"Initializing tokenizer for DataProcessor: {self.filter_config['tokenizer_for_filtering']}")
            try:
                self.tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
                    self.filter_config['tokenizer_for_filtering']
                )
            except Exception as e:
                logging.error(f"Failed to load tokenizer. Error: {e}", exc_info=True)
                raise

    def _normalize_unicode_and_fix_encoding(self, text: str) -> str:
        """Fixes unicode and encoding errors in text using ftfy."""
        return ftfy.fix_text(text)

    def _normalize_structure_markers(self, text: str) -> str:
        """
        Converts structural markers (like SECTION, (a), (i)) into special tokens.

        This method standardizes various legal text formats into a consistent
        representation for the model.
        """
        # Step 1: Standardize "SEC." and "SECTION" headers.
        text = re.sub(
            r'^[ \t]*(SEC|SECTION)\s*\d+\.\s*', '[SECTION_START] ',
            text, flags=re.IGNORECASE | re.MULTILINE
        )
        # Step 2: Convert Roman numeral list markers.
        text = re.sub(
            r'^[ \t]*\(([ivxlcdm]+)\)\s*', lambda m: f'[SUBITEM_{m.group(1).lower()}] ',
            text, flags=re.IGNORECASE | re.MULTILINE
        )
        # Step 3: Convert lettered list markers.
        text = re.sub(
            r'^[ \t]*\(([a-zA-Z])\)\s*', lambda m: f'[SUBSECTION_{m.group(1).lower()}] ',
            text, flags=re.MULTILINE
        )
        # Step 4: Clean up redundant keywords.
        text = re.sub(
            r'\b(subsection|paragraph|item|clause)\s*(?=\[(SUBSECTION_|SUBITEM_)])', '',
            text, flags=re.IGNORECASE
        )
        return text

    def _standardize_whitespace(self, text: str) -> str:
        """
        Replaces multiple newlines with a paragraph break token and normalizes all whitespace.
        """
        # Replace two or more newlines with a special token for paragraph breaks.
        text = re.sub(r'(\n\s*){2,}', ' [PARAGRAPH_BREAK] ', text)
        # Replace remaining single newlines and other whitespace with a single space.
        text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        # Collapse multiple spaces into one.
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _apply_full_cleaning(self, text: str) -> str:
        """
        Applies the entire sequence of cleaning and transformation steps to a single text.

        Args:
            text (str): The raw input string.

        Returns:
            str: The cleaned and transformed string.
        """
        if not isinstance(text, str):
            return ""

        text = self._normalize_unicode_and_fix_encoding(text)



        if self.data_config.get('normalize_structure', False):
            text = self._normalize_structure_markers(text)

        # Apply regex patterns to remove unwanted text segments.
        for pattern in self._REGEX_PATTERNS_TO_REMOVE:
            text = pattern.sub('', text)
            
        text = self._standardize_whitespace(text)
        return text

    def _process_and_enrich_batch(self, examples: Dict[str, List]) -> Dict[str, List]:
        """
        Processes a batch of examples from a Hugging Face Dataset.

        This method applies cleaning, combines titles with text, and adds token counts.
        """
        # Clean the main text column first.
        texts = [self._apply_full_cleaning(text) for text in examples[TEXT_COLUMN]]

        # Process summaries ONLY if the summary column exists
        if SUMMARY_COLUMN in examples:
            summaries = [self._apply_full_cleaning(summary) for summary in examples[SUMMARY_COLUMN]]
            examples[SUMMARY_COLUMN] = summaries
            # Calculate summary token counts correctly.
            examples['summary_token_count'] = [len(self.tokenizer.encode(s)) for s in summaries]

        # Combine title with text if enabled in the config.
        if self.data_config.get('combine_title_with_text', False) and TITLE_COLUMN in examples:
            titles = [title if title else "" for title in examples[TITLE_COLUMN]]
            texts = [f"{title}. {text}".strip() for title, text in zip(titles, texts)]
        
        # Update the final text column.
        examples[TEXT_COLUMN] = texts

        # Calculate text token counts correctly.
        examples['text_token_count'] = [len(self.tokenizer.encode(t)) for t in texts]

        return examples
        
    def _filter_dataset_split(self, dataset_split: Dataset) -> Dataset:
        """Filters a single split of a dataset based on token count limits."""
        if not self.filter_config.get('apply_filtering', False):
            logging.info("Filtering is disabled in the configuration. Skipping.")
            return dataset_split

        min_text = self.filter_config['min_text_tokens']
        max_text = self.filter_config['max_text_tokens']
        min_summary = self.filter_config['min_summary_tokens']
        max_summary = self.filter_config['max_summary_tokens']

        def is_within_limits(example):
            # Check text token count
            text_check = min_text <= example['text_token_count'] <= max_text
            
            # If summary token count exists, check it as well. Otherwise, assume it passes.
            summary_check = True
            if 'summary_token_count' in example:
                summary_check = min_summary <= example['summary_token_count'] <= max_summary
            
            return text_check and summary_check

        original_size = len(dataset_split)
        if 'text_token_count' not in dataset_split.column_names:
             logging.warning("Cannot filter because 'text_token_count' column is missing. Skipping filtering.")
             return dataset_split
        
        filtered_dataset = dataset_split.filter(is_within_limits, num_proc=os.cpu_count())
        new_size = len(filtered_dataset)
        
        logging.info(f"Filtered split: {original_size} -> {new_size} samples. "
                     f"({original_size - new_size} removed).")
        return filtered_dataset

    # --- Public Methods ---

    def process_text(self, text: str, title: str = None) -> str:
        """
        Processes a single raw text string for inference.

        Args:
            text (str): The main body of the text.
            title (str, optional): The title of the text. Defaults to None.

        Returns:
            str: The fully cleaned and prepared text, ready for the model.
        """
        cleaned_text = self._apply_full_cleaning(text)
        if self.data_config.get('combine_title_with_text', False) and title:
            return f"{title}. {cleaned_text}".strip()
        return cleaned_text

    def process_dataset(self, dataset: DatasetDict) -> DatasetDict:
        """
        Applies the full processing and filtering pipeline to a DatasetDict.

        Args:
            dataset (DatasetDict): The raw dataset loaded from files.

        Returns:
            DatasetDict: The processed, filtered, and cleaned dataset.
        """
        logging.info("Starting dataset processing and enrichment...")
        enriched_dataset = dataset.map(
            self._process_and_enrich_batch,
            batched=True,
            num_proc=os.cpu_count()
        )
        
        logging.info("Applying filtering to all dataset splits...")
        filtered_dataset = DatasetDict({
            split: self._filter_dataset_split(enriched_dataset[split])
            for split in enriched_dataset.keys()
        })
        
        # Final column cleanup
        columns_to_keep = {TEXT_COLUMN, SUMMARY_COLUMN}
        first_split_key = next(iter(filtered_dataset))
        columns_to_remove = [
            col for col in filtered_dataset[first_split_key].column_names 
            if col not in columns_to_keep
        ]
        
        final_dataset = filtered_dataset.remove_columns(columns_to_remove)
        logging.info(f"Removed unnecessary columns: {columns_to_remove}")
        
        return final_dataset


def main():
    """
    Main function to run the data processing script from the command line.
    This function demonstrates how to use the DataProcessor class.
    """
    logging.info("--- Starting Standalone Data Processing ---")
    
    # 1. Initialize the processor with configurations
    processor = DataProcessor(
        data_config=DATA_PROCESSING_CONFIG,
        filter_config=FILTERING_CONFIG
    )
    
    # 2. Load the raw dataset
    input_dir = DATA_PROCESSING_CONFIG['source_dir']
    output_dir = DATA_PROCESSING_CONFIG['save_dir']
    
    try:
        data_files = os.path.join(input_dir, 'bill_sum_train.jsonl')
        if not data_files:
            logging.error(f"No .json or .jsonl files found in '{input_dir}'.")
            return
        # Assuming all files belong to the 'train' split when run standalone
        raw_dataset = load_dataset('json', data_files={'train': data_files})
    except Exception as e:
        logging.error(f"Error loading data from '{input_dir}': {e}", exc_info=True)
        return
        
    # 3. Process the dataset using the processor
    processed_dataset = processor.process_dataset(raw_dataset)
    
    # 4. Save the processed data
    os.makedirs(output_dir, exist_ok=True)
    for split_name, dataset_split in processed_dataset.items():
        save_path = os.path.join(output_dir, f'{split_name}.jsonl')
        dataset_split.to_json(save_path, orient='records', lines=True)
        logging.info(f"Successfully saved '{split_name}' split to '{save_path}'")
        
    logging.info("--- Standalone Data Processing Finished ---")


if __name__ == "__main__":
    main()