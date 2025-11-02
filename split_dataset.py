# split_dataset.py

"""
This script splits the processed dataset into training and validation sets.
The test set is omitted as the official BillSum test set will be used for evaluation.
"""

import os
import logging
from datasets import load_dataset, DatasetDict

# Import configurations from the central config file
from config import DATA_SPLIT_CONFIG, LOGGING_CONFIG

# Setup logger
logging.basicConfig(
    level=LOGGING_CONFIG['level'],
    format=LOGGING_CONFIG['format'],
    datefmt=LOGGING_CONFIG['datefmt']
)


def create_train_validation_split(
    processed_data_path: str,
    val_size: float,
    seed: int
) -> DatasetDict:
    """
    Loads the processed data and splits it into training and validation sets.

    Args:
        processed_data_path (str): The path to the .jsonl file of the processed data.
        val_size (float): The proportion of the dataset to allocate for the validation set.
        seed (int): The random seed for shuffling to ensure reproducibility.

    Returns:
        DatasetDict: A dictionary containing the 'train' and 'validation' splits.
    """
    logging.info("Starting data splitting process...")
    try:
        # The processed data is expected to be a single file (e.g., train.jsonl)
        processed_dataset = load_dataset('json', data_files=processed_data_path)['train']
        logging.info(f"Loaded processed data from '{processed_data_path}'. Total samples: {len(processed_dataset)}")
    except FileNotFoundError:
        logging.error(f"Processed data file not found at '{processed_data_path}'.")
        raise
    except Exception as e:
        logging.error(f"An error occurred while loading the data: {e}", exc_info=True)
        raise

    logging.info(f"Splitting data into {1-val_size:.0%} training and {val_size:.0%} validation sets.")
    
    # Perform a single split to create train and validation sets.
    split_result = processed_dataset.train_test_split(
        test_size=val_size,
        shuffle=True,
        seed=seed
    )

    # The result of train_test_split is a DatasetDict with 'train' and 'test' keys.
    # We rename the 'test' key to 'validation' for clarity.
    final_dataset = DatasetDict({
        'train': split_result['train'],
        'validation': split_result['test']
    })

    logging.info("Data splitting complete. Final set sizes:")
    logging.info(f"  - train: {len(final_dataset['train'])} samples")
    logging.info(f"  - validation: {len(final_dataset['validation'])} samples")
    
    return final_dataset


def save_splits(dataset_dict: DatasetDict, output_dir: str):
    """
    Saves the splits from a DatasetDict to the specified directory.

    Args:
        dataset_dict (DatasetDict): The dataset splits to save.
        output_dir (str): The directory where the .jsonl files will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Saving splits to '{output_dir}'...")
    for split_name, dataset_split in dataset_dict.items():
        file_path = os.path.join(output_dir, f'{split_name}.jsonl')
        dataset_split.to_json(file_path, orient='records', lines=True)
        logging.info(f"  -> Saved '{split_name}' split to '{file_path}'")


def main():
    """
    Main function to orchestrate the data splitting process.
    """
    config = DATA_SPLIT_CONFIG
    processed_file_path = os.path.join(config['processed_data_dir'], 'train.jsonl')
    
    # Create the splits
    final_splits = create_train_validation_split(
        processed_data_path=processed_file_path,
        val_size=config['validation_size'],
        seed=config['shuffle_seed']
    )
    
    # Save the splits
    save_splits(final_splits, config['split_data_dir'])
    
    logging.info("Data splitting and saving process completed successfully.")


if __name__ == "__main__":
    main()