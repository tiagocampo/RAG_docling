from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain.tools import Tool
from langchain.tools.retriever import create_retriever_tool
from pydantic import BaseModel, Field
from typing import Any, List
import logging

logger = logging.getLogger(__name__)

class LoggedRetriever(BaseRetriever):
    """A retriever that logs its operations."""
    
    vectorstore: Any = Field(description="The vector store to retrieve from")
    
    class Config:
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Get documents relevant to a query with logging."""
        logger.info(f"Searching documents with query: {query}")
        results = self.vectorstore.similarity_search(query, k=3)
        logger.info(f"Found {len(results)} documents")
        for i, doc in enumerate(results):
            logger.info(f"Document {i + 1}:")
            logger.info(f"Content: {doc.page_content[:200]}...")
            logger.info(f"Metadata: {doc.metadata}")
        return results

def create_retriever_tool(vectorstore: Any) -> Tool:
    """Create a retriever tool with the given vectorstore."""
    retriever = LoggedRetriever(vectorstore=vectorstore)
    
    return create_retriever_tool(
        retriever,
        "search_documents",
        "Search through the uploaded documents to find relevant information."
    ) 