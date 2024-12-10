from typing import Dict, List, Union
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
import logging

logger = logging.getLogger(__name__)

class DocumentService:
    @staticmethod
    def extract_content_and_sources(message: BaseMessage) -> tuple[str, List[Dict]]:
        """Extract content and sources from a message."""
        if hasattr(message, 'content') and isinstance(message.content, (list, tuple)):
            # Handle retriever output (list of documents)
            docs = message.content
            sources = [{
                "page": doc.metadata.get("page", "N/A"),
                "text": doc.page_content,
                "section": doc.metadata.get("section", "N/A")
            } for doc in docs]
            context = "\n\n".join(str(doc.page_content) for doc in docs)
        else:
            # Handle string content
            context = str(message.content)
            sources = []
        
        return context, sources

    @staticmethod
    def get_question_from_messages(messages: List[BaseMessage]) -> str:
        """Extract the question from messages."""
        return messages[0].content if messages else "" 