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
