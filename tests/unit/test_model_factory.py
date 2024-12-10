import pytest
from unittest.mock import patch, MagicMock
from src.models.model_factory import ModelFactory
from src.config.model_config import AVAILABLE_MODELS

@pytest.fixture
def mock_session_state():
    with patch('streamlit.session_state', new_callable=dict) as mock_state:
        mock_state["model_name"] = "Claude 3.5 Sonnet"
        yield mock_state

def test_create_model_anthropic(mock_session_state):
    with patch('os.getenv') as mock_getenv:
        mock_getenv.return_value = "test_api_key"
        model = ModelFactory.create_model()
        assert model is not None
        # Add assertions specific to Anthropic model configuration

def test_create_model_openai(mock_session_state):
    mock_session_state["model_name"] = "GPT-4o"
    model = ModelFactory.create_model()
    assert model is not None
    # Add assertions specific to OpenAI model configuration

def test_create_model_with_explicit_name():
    with patch('os.getenv') as mock_getenv:
        mock_getenv.return_value = "test_api_key"
        model = ModelFactory.create_model("Claude 3.5 Sonnet")
        assert model is not None
        # Add assertions for explicit model name case

def test_create_model_invalid_name():
    with pytest.raises(KeyError):
        ModelFactory.create_model("Invalid Model Name") 