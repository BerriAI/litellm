from typing import List, Optional, Union, Dict, Tuple, Literal
from typing_extensions import TypedDict


class CostPerToken(TypedDict):
    input_cost_per_token: float
    output_cost_per_token: float


class ProviderField(TypedDict):
    field_name: str
    field_type: Literal["string"]
    field_description: str
    field_value: str


class ModelInfo(TypedDict):
    """
    Model info for a given model, this is information found in litellm.model_prices_and_context_window.json
    """

    max_tokens: Optional[int]
    max_input_tokens: Optional[int]
    max_output_tokens: Optional[int]
    input_cost_per_token: float
    output_cost_per_token: float
    litellm_provider: str
    mode: Literal[
        "completion", "embedding", "image_generation", "chat", "audio_transcription"
    ]
    supported_openai_params: Optional[List[str]]
