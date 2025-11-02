# app.py

import logging
from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Import project modules and the new service
from config import (
    EVALUATION_CONFIG, LOGGING_CONFIG, DATA_PROCESSING_CONFIG,
    FILTERING_CONFIG, CHUNKING_CONFIG
)
from processor import DataProcessor
from chunker import Chunker
from summarization_service import SummarizationService

# --- APPLICATION SETUP ---
logging.basicConfig(level=LOGGING_CONFIG['level'], format=LOGGING_CONFIG['format'])
app = Flask(__name__)
service: SummarizationService = None

# --- GLOBAL INITIALIZATION (runs once at startup) ---
def initialize_app():
    """Loads all models and initializes the summarization service."""
    global service
    if service:
        logging.info("Service already initialized.")
        return

    try:
        logging.info("--- Initializing Application: Loading models and components ---")
        
        # Load models and tokenizers from paths specified in config
        chunk_model_path = EVALUATION_CONFIG['chunk_summarizer_model_path']
        synthesis_model_path = EVALUATION_CONFIG['synthesis_model_path']

        chunk_tokenizer = AutoTokenizer.from_pretrained(chunk_model_path)
        chunk_model = AutoModelForSeq2SeqLM.from_pretrained(chunk_model_path)

        synthesis_tokenizer = AutoTokenizer.from_pretrained(synthesis_model_path)
        synthesis_model = AutoModelForSeq2SeqLM.from_pretrained(synthesis_model_path)
        
        # Initialize helper classes
        processor = DataProcessor(DATA_PROCESSING_CONFIG, FILTERING_CONFIG, tokenizer=chunk_tokenizer)
        chunker = Chunker(CHUNKING_CONFIG, tokenizer=chunk_tokenizer)

        # Inject all dependencies into the service
        service = SummarizationService(
            config=EVALUATION_CONFIG,
            processor=processor,
            chunker=chunker,
            chunk_model=chunk_model,
            chunk_tokenizer=chunk_tokenizer,
            synthesis_model=synthesis_model,
            synthesis_tokenizer=synthesis_tokenizer
        )
        
        logging.info("--- Application Initialization Complete ---")

    except Exception as e:
        logging.critical(f"A critical error occurred during model loading: {e}", exc_info=True)
        # If models fail to load, the service will remain None, and the API will return an error.

# --- API ROUTES ---

@app.route('/')
def index():
    """Renders the main user interface."""
    return render_template('index.html')


@app.route('/summarize', methods=['POST'])
def handle_summarize_request():
    """
    Handles the summarization request from the client.
    Delegates the core logic to the SummarizationService.
    """
    if not service:
        return jsonify({
            "error": "The summarization service is not available due to a startup error. Please check the server logs."
        }), 503 # Service Unavailable

    try:
        data = request.get_json()
        raw_text = data.get('text')

        if not raw_text or not raw_text.strip():
            return jsonify({"error": "Please provide text to summarize."}), 400

        summary = service.summarize(raw_text)
        return jsonify({"summary": summary})

    except ValueError as ve:
        logging.warning(f"Validation error: {ve}")
        return jsonify({"error": str(ve)}), 413
    except Exception as e:
        logging.error(f"An unexpected error occurred during summarization: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred while processing your request."}), 500


if __name__ == '__main__':
    initialize_app()
    # For production, use a WSGI server like Gunicorn or Waitress instead of app.run()
    app.run(host='0.0.0.0', port=5000, debug=False)