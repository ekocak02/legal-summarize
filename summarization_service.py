# summarization_service.py

import logging
import torch
from typing import List, Dict, Any

# Assuming original project modules are available
from processor import DataProcessor
from chunker import Chunker
from config import APP_CONFIG

# Suppress verbose logging from transformers
logging.getLogger("transformers").setLevel(logging.ERROR)


class SummarizationService:
    """
    Encapsulates the entire hierarchical summarization pipeline.

    This service is responsible for loading all necessary models and components,
    and processing raw text to generate a final, synthesized summary. It is
    designed to be framework-agnostic.
    """

    def __init__(self, config: Dict[str, Any], processor: DataProcessor, chunker: Chunker,
                 chunk_model: Any, chunk_tokenizer: Any,
                 synthesis_model: Any, synthesis_tokenizer: Any):
        """
        Initializes the SummarizationService with all its dependencies.

        Args:
            config (Dict[str, Any]): The evaluation configuration dictionary.
            processor (DataProcessor): An instance of the data processor.
            chunker (Chunker): An instance of the text chunker.
            chunk_model: The pre-loaded chunk summarizer model.
            chunk_tokenizer: The pre-loaded tokenizer for the chunk model.
            synthesis_model: The pre-loaded synthesis model.
            synthesis_tokenizer: The pre-loaded tokenizer for the synthesis model.
        """
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Injected dependencies
        self.processor = processor
        self.chunker = chunker
        self.chunk_model = chunk_model.to(self.device)
        self.chunk_tokenizer = chunk_tokenizer
        self.synthesis_model = synthesis_model.to(self.device)
        self.synthesis_tokenizer = synthesis_tokenizer

        # Set models to evaluation mode
        self.chunk_model.eval()
        self.synthesis_model.eval()
        
        logging.info(f"SummarizationService initialized on device: '{self.device}'")

    def _generate_summaries_batch(self, texts: List[str], model: Any, tokenizer: Any,
                                  gen_params: Dict[str, Any]) -> List[str]:
        """
        Helper function to generate summaries for a batch of texts.

        Args:
            texts (List[str]): A list of texts to summarize.
            model: The sequence-to-sequence model to use.
            tokenizer: The tokenizer corresponding to the model.
            gen_params (Dict[str, Any]): Generation parameters (e.g., num_beams).

        Returns:
            List[str]: A list of generated summaries.
        """
        inputs = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=1024
        ).to(self.device)

        with torch.no_grad():
            summary_ids = model.generate(**inputs, **gen_params)
            
        return tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
    
    def _run_synthesis_step(self, text_to_synthesize: str, prompt_template: str, gen_params: Dict) -> str:
        """Helper function to run a single synthesis step with a given prompt."""
        prompt = prompt_template.format(chunk_summaries=text_to_synthesize)
        return self._generate_summaries_batch([prompt], self.synthesis_model, self.synthesis_tokenizer, gen_params)[0]


    def summarize(self, raw_text: str) -> str:
        """
        Executes the full end-to-end summarization pipeline, including conditional
        intermediate synthesis for very long documents.
        """
        # Step 0: Validate input length
        char_limit = APP_CONFIG.get('character_limit', 45000)
        if len(raw_text) > char_limit:
            raise ValueError(f"Input text is too long. The maximum allowed length is {char_limit} characters.")
        if not raw_text or not raw_text.strip():
            raise ValueError("Input text cannot be empty.")

        # Step 1: Clean and chunk the source text
        logging.info("Step 1/4: Cleaning and chunking text...")
        cleaned_text = self.processor.process_text(raw_text)
        chunks = self.chunker.chunk_for_inference(cleaned_text)

        if not chunks:
            return "The provided text is too short to require hierarchical summarization."

        prompt_for_bart = "Summarize the following section of a legal document: "
        chunks_with_prompt = [prompt_for_bart + chunk for chunk in chunks]

        # Step 2: Summarize each chunk with the first model (e.g., BART)
        logging.info(f"Step 2/4: Summarizing {len(chunks)} chunks...")
        chunk_summaries = self._generate_summaries_batch(
            chunks_with_prompt, # Use 'chunks_with_prompt' instead of just 'chunks'
            self.chunk_model, self.chunk_tokenizer, self.config['chunk_generation_params']
        )
        combined_chunk_summaries = " ".join(chunk_summaries)

        # Step 3: Conditional, multi-layer synthesis
        logging.info("Step 3/4: Preparing for synthesis...")
        synthesizer_input_ids = self.synthesis_tokenizer(combined_chunk_summaries, return_tensors="pt").input_ids
        final_input_text = combined_chunk_summaries
        
        # Check if the combined chunk summaries are too long for the synthesizer
        if synthesizer_input_ids.shape[1] > self.config['synthesizer_input_max_tokens']:
            logging.info(f"Combined summaries ({synthesizer_input_ids.shape[1]} tokens) are too long. Applying intermediate synthesis layer.")
            
            # Re-chunk the combined summaries themselves
            summary_chunks = self.chunker.chunk_for_inference(combined_chunk_summaries)
            
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
            logging.info("Combined summaries are short enough for a single synthesis step.")

        # Step 4: Final synthesis step
        logging.info("Step 4/4: Generating final summary...")
        generated_summary = self._run_synthesis_step(
            final_input_text,
            self.config['final_synthesis_prompt'],
            self.config['final_synthesis_generation_params']
        )
        
        logging.info("Summarization pipeline completed successfully.")
        return generated_summary