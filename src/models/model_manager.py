import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# Define available models and their configurations
AVAILABLE_MODELS = {
    "Claude 3.5 Sonnet": {
        "provider": "anthropic",
        "name": "claude-3-5-sonnet-20241022",
        "temperature": 0.1,
        "streaming": True,
        "max_tokens": 8192,
        "description": "Most intelligent Anthropic model, best for complex tasks and deep analysis"
    },
    "Claude 3.5 Haiku": {
        "provider": "anthropic",
        "name": "claude-3-5-haiku-20241022",
        "temperature": 0.1,
        "streaming": True,
        "max_tokens": 8192,
        "description": "Fast Anthropic model, great for quick responses while maintaining quality"
    },
    "GPT-4o": {
        "provider": "openai",
        "name": "gpt-4o-2024-11-20",
        "temperature": 0.1,
        "streaming": True,
        "max_tokens": 16384,
        "description": "OpenAI's most advanced model, 2x faster than GPT-4 Turbo with multimodal capabilities"
    },
    "GPT-4o Mini": {
        "provider": "openai",
        "name": "gpt-4o-mini",
        "temperature": 0.1,
        "streaming": True,
        "max_tokens": 16384,
        "description": "OpenAI's affordable and intelligent small model, more capable than GPT-3.5 Turbo"
    }
}

# Default model
DEFAULT_MODEL = "GPT-4o"

def get_model():
    """Get the model based on configuration."""
    selected_model = st.session_state.get("model_name", DEFAULT_MODEL)
    model_config = AVAILABLE_MODELS[selected_model]
    
    if model_config["provider"] == "anthropic":
        return ChatAnthropic(
            model=model_config["name"],
            temperature=model_config["temperature"],
            max_tokens=model_config["max_tokens"],
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            streaming=model_config["streaming"]
        )
    else:  # OpenAI
        return ChatOpenAI(
            model=model_config["name"],
            temperature=model_config["temperature"],
            max_tokens=model_config["max_tokens"],
            streaming=model_config["streaming"]
        ) 