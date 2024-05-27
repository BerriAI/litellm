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

    max_tokens: int
    max_input_tokens: int
    max_output_tokens: int
    input_cost_per_token: float
    output_cost_per_token: float
    litellm_provider: str
    mode: str
    supported_openai_params: List[str]
