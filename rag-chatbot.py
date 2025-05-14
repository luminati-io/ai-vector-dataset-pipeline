import os
from sentence_transformers import SentenceTransformer # For generating query embeddings
from pinecone import Pinecone # Official Pinecone client
import google.generativeai as genai # For the generative LLM
from dotenv import load_dotenv

# --- Configuration ---
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("walmart_rag_pipeline")

# Load environment variables from .env file
load_dotenv()

# Pinecone Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "brightdata-ai-dataset"  # Your existing index

# Embedding Model Configuration (MUST BE THE SAME AS USED FOR INDEXING)
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

# Google Gemini API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = 'gemini-2.0-flash'

# --- Initialize Services ---
pinecone_index_instance = None
sentence_model_instance = None
gemini_model_instance = None

def initialize_all_services():
    """Initializes Pinecone, SentenceTransformer, and Gemini models."""
    global pinecone_index_instance, sentence_model_instance, gemini_model_instance

    # Initialize Pinecone
    if not PINECONE_API_KEY:
        logger.critical("🔴 CRITICAL: PINECONE_API_KEY is not set. Please update it in .env or script.")
        return False
    try:
        logger.info("🌲 Initializing Pinecone connection...")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        pinecone_index_instance = pc.Index(PINECONE_INDEX_NAME)
        logger.info(f"✅ Connected to Pinecone index '{PINECONE_INDEX_NAME}'. Stats: {pinecone_index_instance.describe_index_stats()}")
    except Exception as e:
        logger.critical(f"🔴 CRITICAL: Could not initialize Pinecone or connect to index '{PINECONE_INDEX_NAME}': {e}")
        return False

    # Initialize Sentence Transformer Embedding Model
    try:
        logger.info(f"🧠 Initializing embedding model: {EMBEDDING_MODEL_NAME}...")
        sentence_model_instance = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.info("✅ Embedding model initialized.")
    except Exception as e:
        logger.critical(f"🔴 CRITICAL: Failed to load SentenceTransformer model '{EMBEDDING_MODEL_NAME}': {e}")
        return False
        
    # Initialize Google Gemini Model
    if not GEMINI_API_KEY:
        logger.critical("🔴 CRITICAL: GEMINI_API_KEY is not set. Please update it in .env or script.")
        return False
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model_instance = genai.GenerativeModel(GEMINI_MODEL_NAME)
        logger.info(f"✨ Initialized Gemini model: {GEMINI_MODEL_NAME}")
    except Exception as e:
        logger.critical(f"🔴 CRITICAL: Failed to initialize Gemini model: {e}")
        return False
        
    return True

def retrieve_relevant_reviews(query_text, top_k=3):
    """Retrieves relevant reviews from Pinecone using semantic search."""
    if not pinecone_index_instance or not sentence_model_instance:
        logger.error("🔴 Services not initialized. Cannot retrieve reviews.")
        return []
    if not query_text or not query_text.strip():
        logger.warning("⚠️ Query text is empty for retrieval.")
        return []

    logger.info(f"🔎 Retrieving top {top_k} relevant reviews for query: '{query_text}'")
    try:
        query_embedding = sentence_model_instance.encode(query_text).tolist()
        query_response = pinecone_index_instance.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        matches = query_response.get('matches', [])
        logger.info(f"   Found {len(matches)} relevant reviews from Pinecone.")
        return matches
    except Exception as e:
        logger.error(f"🔴 ERROR during review retrieval from Pinecone: {e}", exc_info=True)
        return []

def generate_answer_with_rag(user_query, retrieved_reviews):
    """Generates an answer using Gemini, augmented with retrieved review context."""
    if not gemini_model_instance:
        logger.error("🔴 Gemini model not initialized. Cannot generate answer.")
        return "Error: Generative model not available."
    if not user_query or not user_query.strip():
        logger.warning("⚠️ User query is empty for RAG.")
        return "Please provide a query."

    if not retrieved_reviews:
        logger.info("ℹ️ No specific reviews found to augment the prompt. Using general knowledge (or a fallback response).")
        # Fallback: Ask Gemini directly without context, or return a message.
        # For this example, let's try to answer directly but state no context was found.
        context_str = "No specific reviews were found in the knowledge base to directly answer this."
    else:
        logger.info(f"📝 Augmenting prompt with {len(retrieved_reviews)} reviews.")
        context_parts = []
        for i, match in enumerate(retrieved_reviews):
            metadata = match.get('metadata', {})
            title = metadata.get('title', '')
            description = metadata.get('description', 'No description available.')
            context_parts.append(f"Review {i+1} (Score: {match.score:.2f}):\nTitle: {title}\nDescription: {description}\n---")
        context_str = "\n".join(context_parts)

    prompt = f"""
    You are a helpful assistant analyzing customer reviews from Walmart for a specific product.
    Based on the following retrieved customer review(s), please answer the user's question.
    If the reviews don't directly answer the question, say that the provided reviews don't contain specific information on that topic, but you can still try to answer generally if possible.

    Retrieved Review(s):
    ---
    {context_str}
    ---

    User's Question: {user_query}

    Answer:
    """
    logger.info(f"💬 Sending augmented prompt to Gemini for question: '{user_query}'")
    # logger.debug(f"Full prompt to Gemini:\n{prompt}") # Uncomment to see the full prompt

    try:
        response = gemini_model_instance.generate_content(prompt)
        logger.info("✅ Received response from Gemini.")
        return response.text
    except Exception as e:
        logger.error(f"🔴 ERROR during answer generation with Gemini: {e}", exc_info=True)
        # Check for specific feedback if the error is from the API (e.g., content filtering)
        if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
             logger.error(f"   Gemini Prompt Feedback: {response.prompt_feedback}")
        return "Sorry, I encountered an error trying to generate an answer."

if __name__ == "__main__":
    logger.info("🚀 Starting RAG Pipeline for Walmart Reviews...")

    if not initialize_all_services():
        logger.critical("🔴 Failed to initialize one or more services. Exiting application.")
        exit()

    # Example User Queries for RAG
    user_questions = [
        "What do people generally say about the price of this MacBook Air?",
        "Is the battery life good enough for a college student?",
        "Are there any common complaints about shipping or delivery for this item?",
        "Is this laptop considered lightweight and good for travel based on reviews?",
        "How does this MacBook Air compare to Chromebooks, according to customer experiences?",
        "Tell me about the screen quality."
    ]

    for question in user_questions:
        print(f"\n\n{'='*20} User Question: {question} {'='*20}")
        
        # 1. Retrieve relevant reviews
        relevant_reviews = retrieve_relevant_reviews(question, top_k=3) # Get top 3 reviews

        # 2. Generate answer using RAG
        answer = generate_answer_with_rag(question, relevant_reviews)
        
        print(f"\n🤖 Assistant's Answer:\n{answer}")

    logger.info("\n🎉 RAG Pipeline Demonstration Finished.")
