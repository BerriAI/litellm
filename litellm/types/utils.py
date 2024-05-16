from typing import List, Optional, Union, Dict, Tuple, Literal, TypedDict


class CostPerToken(TypedDict):
    input_cost_per_token: float
    output_cost_per_token: float
