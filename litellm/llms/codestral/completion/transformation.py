import json
from typing import Optional

import litellm
from litellm.llms.openai.completion.transformation import OpenAITextCompletionConfig
from litellm.types.llms.databricks import GenericStreamingChunk


class CodestralTextCompletionConfig(OpenAITextCompletionConfig):
    """
    Reference: https://docs.mistral.ai/api/#operation/createFIMCompletion
    """

    suffix: Optional[str] = None
    temperature: Optional[int] = None
    max_tokens: Optional[int] = None
    min_tokens: Optional[int] = None
    stream: Optional[bool] = None
    random_seed: Optional[int] = None

    def __init__(
        self,
        suffix: Optional[str] = None,
        temperature: Optional[int] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        min_tokens: Optional[int] = None,
        stream: Optional[bool] = None,
        random_seed: Optional[int] = None,
        stop: Optional[str] = None,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return super().get_config()

    def get_supported_openai_params(self, model: str):
        return [
            "suffix",
            "temperature",
            "top_p",
            "max_tokens",
            "max_completion_tokens",
            "stream",
            "seed",
            "stop",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        for param, value in non_default_params.items():
            if param == "suffix":
                optional_params["suffix"] = value
            if param == "temperature":
                optional_params["temperature"] = value
            if param == "top_p":
                optional_params["top_p"] = value
            if param == "max_tokens" or param == "max_completion_tokens":
                optional_params["max_tokens"] = value
            if param == "stream" and value is True:
                optional_params["stream"] = value
            if param == "stop":
                optional_params["stop"] = value
            if param == "seed":
                optional_params["random_seed"] = value
            if param == "min_tokens":
                optional_params["min_tokens"] = value

        return optional_params

    def _chunk_parser(self, chunk_data: str) -> GenericStreamingChunk:
        text = ""
        is_finished = False
        finish_reason = None
        logprobs = None

        chunk_data = (
            litellm.CustomStreamWrapper._strip_sse_data_from_chunk(chunk_data) or ""
        )
        chunk_data = chunk_data.strip()
        if len(chunk_data) == 0 or chunk_data == "[DONE]":
            return {
                "text": "",
                "is_finished": is_finished,
                "finish_reason": finish_reason,
            }
        try:
            chunk_data_dict = json.loads(chunk_data)
        except json.JSONDecodeError:
            return {
                "text": "",
                "is_finished": is_finished,
                "finish_reason": finish_reason,
            }

        original_chunk = litellm.ModelResponse(**chunk_data_dict, stream=True)
        _choices = chunk_data_dict.get("choices", []) or []
        _choice = _choices[0]
        text = _choice.get("delta", {}).get("content", "")

        if _choice.get("finish_reason") is not None:
            is_finished = True
            finish_reason = _choice.get("finish_reason")
            logprobs = _choice.get("logprobs")

        return GenericStreamingChunk(
            text=text,
            original_chunk=original_chunk,
            is_finished=is_finished,
            finish_reason=finish_reason,
            logprobs=logprobs,
        )
