from typing import TypedDict, Optional

class ModelConfig(TypedDict):
    provider: str
    name: str
    temperature: float
    streaming: bool
    max_tokens: int
    description: str

AVAILABLE_MODELS: dict[str, ModelConfig] = {
    "Claude 3.5 Sonnet": {
        "provider": "anthropic",
        "name": "claude-3-5-sonnet-20241022",
        "temperature": 0.1,
        "streaming": True,
        "max_tokens": 8192,
        "description": "Most intelligent Anthropic model, best for complex tasks and deep analysis"
    },
    # ... other models ...
}

DEFAULT_MODEL = "GPT-4o" 