from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import streamlit as st
import os
import tempfile
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global client instance
_client = None

def get_vectorstore():
    """Initialize or get the Qdrant vector store."""
    
    # Check for API key
    openai_key = st.session_state.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError("OpenAI API key is required for embeddings. Please enter it in the sidebar.")
    
    global _client
    
    # Create a unique path for Qdrant storage in the system's temp directory
    storage_path = os.path.join(tempfile.gettempdir(), "qdrant_storage")
    os.makedirs(storage_path, exist_ok=True)
    
    # Initialize client if not exists
    if _client is None:
        try:
            _client = QdrantClient(
                path=storage_path,
                prefer_grpc=True
            )
            
            # Initialize collection if it doesn't exist
            collections = _client.get_collections().collections
            if not any(collection.name == "documents" for collection in collections):
                _client.create_collection(
                    collection_name="documents",
                    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
                )
                logger.info("Created new Qdrant collection: documents")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {str(e)}")
            raise
    
    # Create embeddings with API key
    embedding_model = OpenAIEmbeddings(
        openai_api_key=openai_key
    )
    
    return QdrantVectorStore(
        client=_client,
        collection_name="documents",
        embedding=embedding_model,
    ) 