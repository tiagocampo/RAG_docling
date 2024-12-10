import pytest
from unittest.mock import Mock, patch
from langchain_core.messages import HumanMessage, AIMessage
from src.services.chat_node_service import ChatNodeService
from src.services.document_service import DocumentService

@pytest.fixture
def mock_model():
    model = Mock()
    model.invoke.return_value = AIMessage(content="Test response")
    return model

@pytest.fixture
def chat_service():
    with patch('src.services.chat_node_service.ModelFactory') as mock_factory:
        mock_factory.create_model.return_value = mock_model()
        service = ChatNodeService()
        yield service

def test_process_agent(chat_service):
    messages = [HumanMessage(content="test question")]
    result = chat_service.process_agent(messages)
    
    assert "messages" in result
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)

def test_process_rewrite(chat_service):
    messages = [HumanMessage(content="test question")]
    result = chat_service.process_rewrite(messages)
    
    assert "messages" in result
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)

@pytest.fixture
def document_service():
    return DocumentService()

def test_extract_content_and_sources(document_service):
    message = AIMessage(content=[
        Mock(page_content="test content", metadata={"page": 1, "section": "intro"})
    ])
    
    context, sources = document_service.extract_content_and_sources(message)
    
    assert context == "test content"
    assert len(sources) == 1
    assert sources[0]["page"] == 1
    assert sources[0]["section"] == "intro" 