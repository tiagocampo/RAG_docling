import pytest
from unittest.mock import Mock, patch
from src.models.retriever import LoggedRetriever, create_retriever_tool
from langchain_core.documents import Document

@pytest.fixture
def mock_vectorstore():
    vectorstore = Mock()
    vectorstore.similarity_search.return_value = [
        Document(page_content="Test content", metadata={"page": 1})
    ]
    return vectorstore

def test_logged_retriever_get_documents(mock_vectorstore):
    retriever = LoggedRetriever(vectorstore=mock_vectorstore)
    docs = retriever._get_relevant_documents("test query")
    
    assert len(docs) == 1
    assert docs[0].page_content == "Test content"
    mock_vectorstore.similarity_search.assert_called_once_with("test query", k=3)

def test_create_retriever_tool(mock_vectorstore):
    tool = create_retriever_tool(mock_vectorstore)
    
    assert tool.name == "search_documents"
    assert "Search through the uploaded documents" in tool.description 