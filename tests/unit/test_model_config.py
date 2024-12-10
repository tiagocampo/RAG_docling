import pytest
from src.config.model_config import AVAILABLE_MODELS, DEFAULT_MODEL

def test_available_models_structure():
    assert isinstance(AVAILABLE_MODELS, dict)
    for model_name, config in AVAILABLE_MODELS.items():
        assert isinstance(model_name, str)
        assert isinstance(config, dict)
        assert "provider" in config
        assert "name" in config
        assert "temperature" in config
        assert "streaming" in config
        assert "max_tokens" in config
        assert "description" in config

def test_default_model_exists():
    assert DEFAULT_MODEL in AVAILABLE_MODELS

def test_model_providers():
    providers = {config["provider"] for config in AVAILABLE_MODELS.values()}
    assert providers.issubset({"anthropic", "openai"})

def test_model_parameters():
    for config in AVAILABLE_MODELS.values():
        assert 0 <= config["temperature"] <= 1
        assert isinstance(config["streaming"], bool)
        assert isinstance(config["max_tokens"], int)
        assert config["max_tokens"] > 0 