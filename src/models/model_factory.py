import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from src.config.model_config import AVAILABLE_MODELS, DEFAULT_MODEL

class ModelFactory:
    @staticmethod
    def create_model(model_name: str = None):
        """Create and return an LLM instance based on the model name."""
        selected_model = model_name or st.session_state.get("model_name", DEFAULT_MODEL)
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