Hierarchical Legal Text Summarization (BART & Flan-T5)

This project implements an end-to-end hierarchical summarization pipeline designed specifically for long legal documents. It uses a two-stage approach: a BART-based chunk summarizer generates intermediate summaries, and a Flan-T5-based synthesizer combines these into a final, coherent summary.

The entire pipeline is wrapped in a Flask API for easy integration and use.

🚀 Features

Hierarchical Pipeline: Handles texts longer than a model's token limit by chunking, summarizing chunks, and synthesizing summaries.

Hybrid Model Approach: Uses BART for detailed intermediate summarization and Flan-T5 for high-level final synthesis.

Modular & Testable: Code is structured into decoupled services (DataProcessor, Chunker, SummarizationService) for clarity and maintainability.

Configuration Driven: All model paths, training hyperparameters, and processing settings are managed via a central config.py.

Flask API: Includes an app.py to serve the model as a web service.

Data & Model Versioning: Uses DVC to track large data and model files alongside Git.

🏛️ Architecture & Pipeline

The summarization logic follows a multi-step process:

Process: The raw text is cleaned, normalized (e.g., fixing unicode, standardizing legal markers) by the DataProcessor.

Chunk (BART): The cleaned text is intelligently split into overlapping semantic chunks by the Chunker.

Summarize Chunks: Each chunk is summarized individually by the fine-tuned BART model ("Chunk Summarizer").

Combine & Check: The intermediate summaries are combined into a single document.

Synthesize (Flan-T5):

If Short: The combined text is sent directly to the Flan-T5 model ("Synthesizer").

If Long: The combined text is re-chunked and passes through an intermediate synthesis layer before the final synthesis step.

Final Summary: The Flan-T5 model produces the final, comprehensive summary.

🔧 Installation

Clone the repository:

git clone [https://github.com/ekocak02/legal-summarize.git](https://github.com/ekocak02/legal-summarize.git)
cd legal-summarize



Create and activate a virtual environment (recommended):

python3 -m venv venv
source venv/bin/activate



Install the required packages:

pip install -r requirements.txt



Download models and data using DVC:

dvc pull



Download NLTK data (required for ROUGE scoring and sentence tokenization):

python
'''
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
'''

🏃‍♂️ How to Run

1. Training the Pipeline

The pipeline requires two models to be trained in sequence.

Step 1: Train the BART Chunk Summarizer
This model learns to summarize small text chunks.

python train_bart.py



Step 2: Generate Data for the Synthesizer
This script uses the trained BART model to create a new training set for T5.

python generate_synthesizer_data.py



Step 3: Train the Flan-T5 Synthesizer
This model learns to "summarize the summaries" into a final, coherent text.

python train_flanT5.py



2. Evaluating the Model

To run the full pipeline on the test set and save the predictions:

python evaluation.py



To calculate ROUGE and BERTScore metrics from the generated predictions (ensure you have results/final_evaluation_predictions.jsonl or update the path):

python evaluate_scores.py results/final_evaluation_predictions.jsonl



3. Running the Web Application (API)

Once models are trained and paths are set in config.py, you can run the Flask server.

Note: This runs the built-in Flask development server, which is not suitable for production use.

python app.py



The server will start on http://0.0.0.0:5000. You can access the simple UI at this address or send a POST request to /summarize:

Request:

{
  "text": "Your very long legal document text goes here..."
}



Response:

{
  "summary": "The final, synthesized summary of your document."
}



📈 Evaluation Results

The following scores were achieved on the BillSum test set using the final hierarchical model (BART + Flan-T5) by running python evaluate_scores.py.

Metric                      Score

ROUGE-1                     50.8791

ROUGE-2                     28.2616

ROUGE-L                     34.4282

BERTScore F1                87.4414

BERTScore Precision         88.4245

BERTScore Recall            86.5346

📂 Project Structure

├── models/                     # Trained model checkpoints (tracked by DVC)
├── data/                       # Raw, processed, and split data (tracked by DVC)
├── results/                    # Training checkpoints and evaluation outputs
├── templates/                  # HTML template for the Flask app
│
├── app.py                      # Flask API server
├── summarization_service.py    # Core summarization logic
│
├── config.py                   # Central configuration file
├── processor.py                # Text cleaning and processing class
├── chunker.py                  # Text chunking and oracle summary class
│
├── train_bart.py               # Script for training Model 1
├── generate_synthesizer_data.py # Script to generate data for Model 2
├── train_flanT5.py             # Script for training Model 2
│
├── evaluation.py               # Run inference on test set
├── evaluate_scores.py          # Calculate ROUGE/BERTScore metrics
│
├── requirements.txt            # Project dependencies (with pinned versions)
├── data.dvc                    # DVC file for /data
├── models.dvc                  # DVC file for /models
└── README.md                   # This file



🌱 Future Improvements

This project provides a solid foundation. Future work could include:

Unit Testing: Implementing unit tests (e.g., using pytest) for the processor.py and chunker.py modules to ensure robustness and prevent regressions.

Dockerization: Creating a Dockerfile and docker-compose.yml to containerize the application (including DVC setup) for easy, reproducible deployment.