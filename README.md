# Walmart Review Analysis and Retrieval System

This repository provides tools to scrape, process, and analyze customer reviews from Walmart. The system leverages Pinecone for vector storage, Sentence Transformers for semantic search, and Google's Gemini AI for review extraction and analysis.

## Components

- **`rag-chatbot.py`**: Main script for initializing services and generating responses using a Retrieval-Augmented Generator (RAG) pipeline.
- **`review_to_pinecone.py`**: Processes extracted reviews, generates embeddings, and stores them in Pinecone.
- **`semantic_search_client.py`**: Provides a service to perform semantic search queries on stored embeddings.
- **`walmart_review_extractor.py`**: Uses Bright Data to scrape reviews from Walmart and processes them using Google Gemini AI.

## Prerequisites

- Python 3.7+
- Access keys for Pinecone, Gemini AI, and Bright Data.

## Setup

1. Clone the repository:
   - `git clone https://github.com/yourusername/your-repo.git`
   - `cd your-repo`

2. Install the required Python packages:
   - `pip install -r requirements.txt`

3. Set up environment variables by creating a `.env` file in the root directory. Add the following variables:
   - `PINECONE_API_KEY=your_pinecone_api_key`
   - `GEMINI_API_KEY=your_gemini_api_key`
   - `BRIGHT_DATA_API_KEY=your_bright_data_api_key`

4. Place your reviews dataset file (`extracted_walmart_reviews.json`) in the project root for processing.

## Usage

### Extract Reviews

Use `walmart_review_extractor.py` to fetch reviews from Walmart:
- `python walmart_review_extractor.py`

### Process Reviews and Populate Pinecone

Process the extracted reviews and upsert them into Pinecone using:
- `python review_to_pinecone.py`

### Semantic Search

Perform semantic searches on your indexed data using:
- `python semantic_search_client.py`

### Run the RAG Chatbot

Finally, to run the RAG pipeline and interact with it:
- `python rag-chatbot.py`

## Logging

Logs are saved in `data_pipeline.log` and also printed to the console for real-time monitoring.

## Note

Ensure that all the required environment variables and API keys are configured correctly for seamless operation of the scripts.
