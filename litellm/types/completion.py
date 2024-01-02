from typing import List, Optional, Union

from pydantic import BaseModel, validator


class CompletionRequest(BaseModel):
    model: str
    messages: List[str] = []
    timeout: Optional[Union[float, int]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    stream: Optional[bool] = None
    stop: Optional[dict] = None
    max_tokens: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[dict] = None
    user: Optional[str] = None
    response_format: Optional[dict] = None
    seed: Optional[int] = None
    tools: Optional[List[str]] = None
    tool_choice: Optional[str] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    deployment_id: Optional[str] = None
    functions: Optional[List[str]] = None
    function_call: Optional[str] = None
    base_url: Optional[str] = None
    api_version: Optional[str] = None
    api_key: Optional[str] = None
    model_list: Optional[List[str]] = None

    class Config:
        # allow kwargs
        extra = "allow"
