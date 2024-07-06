from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# from here: https://docs.rungalileo.io/galileo/gen-ai-studio-products/galileo-observe/how-to/logging-data-via-restful-apis#structuring-your-records
class LLMResponse(BaseModel):
    latency_ms: int
    status_code: int
    input_text: str
    output_text: str
    node_type: str
    model: str
    num_input_tokens: int
    num_output_tokens: int
    output_logprobs: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional. When available, logprobs are used to compute Uncertainty.",
    )
    created_at: str = Field(
        ..., description='timestamp constructed in "%Y-%m-%dT%H:%M:%S" format'
    )
    tags: Optional[List[str]] = None
    user_metadata: Optional[Dict[str, Any]] = None
