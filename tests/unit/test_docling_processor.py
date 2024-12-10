import pytest
from unittest.mock import Mock, patch
from src.utils.docling_processor import DoclingProcessor

@pytest.fixture
def mock_model():
    model = Mock()
    model.invoke.return_value = "Test response"
    return model

@pytest.fixture
def processor():
    with patch('src.models.model_factory.ModelFactory') as mock_factory:
        mock_factory.create_model.return_value = mock_model()
        processor = DoclingProcessor()
        yield processor

def test_docling_processor_initialization(processor):
    assert processor.model is not None

def test_process_document(processor, tmp_path):
    # Create a temporary test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("Test content")
    
    result = processor.process_document(str(test_file))
    assert isinstance(result, dict)
    # Add more specific assertions based on your expected output structure 