"""
Nvidia NIM endpoint: https://docs.api.nvidia.com/nim/reference/databricks-dbrx-instruct-infer

This is OpenAI compatible

This file only contains param mapping logic

API calling is done using the OpenAI SDK with an api_base
"""
from typing import Any, List, Optional, cast

import httpx

from litellm.litellm_core_utils.prompt_templates.common_utils import (
    _parse_content_for_reasoning,
)
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.types.utils import ModelResponse


class NvidiaNimConfig(OpenAIGPTConfig):
    """
    Reference: https://docs.api.nvidia.com/nim/reference/databricks-dbrx-instruct-infer

    The class `NvidiaNimConfig` provides configuration for the Nvidia NIM's Chat Completions API interface. Below are the parameters:
    """

    def get_supported_openai_params(self, model: str) -> list:
        """
        Get the supported OpenAI params for the given model


        Updated on July 5th, 2024 - based on https://docs.api.nvidia.com/nim/reference
        """
        if model in [
            "google/recurrentgemma-2b",
            "google/gemma-2-27b-it",
            "google/gemma-2-9b-it",
            "gemma-2-9b-it",
        ]:
            return ["stream", "temperature", "top_p", "max_tokens", "stop", "seed"]
        elif model == "nvidia/nemotron-4-340b-instruct":
            return [
                "stream",
                "temperature",
                "top_p",
                "max_tokens",
                "max_completion_tokens",
            ]
        elif model == "nvidia/nemotron-4-340b-reward":
            return [
                "stream",
            ]
        elif model in ["google/codegemma-1.1-7b"]:
            # most params - but no 'seed' :(
            return [
                "stream",
                "temperature",
                "top_p",
                "frequency_penalty",
                "presence_penalty",
                "max_tokens",
                "max_completion_tokens",
                "stop",
            ]
        else:
            # DEFAULT Case - The vast majority of Nvidia NIM Models lie here
            # "upstage/solar-10.7b-instruct",
            # "snowflake/arctic",
            # "seallms/seallm-7b-v2.5",
            # "nvidia/llama3-chatqa-1.5-8b",
            # "nvidia/llama3-chatqa-1.5-70b",
            # "mistralai/mistral-large",
            # "mistralai/mixtral-8x22b-instruct-v0.1",
            # "mistralai/mixtral-8x7b-instruct-v0.1",
            # "mistralai/mistral-7b-instruct-v0.3",
            # "mistralai/mistral-7b-instruct-v0.2",
            # "mistralai/codestral-22b-instruct-v0.1",
            # "microsoft/phi-3-small-8k-instruct",
            # "microsoft/phi-3-small-128k-instruct",
            # "microsoft/phi-3-mini-4k-instruct",
            # "microsoft/phi-3-mini-128k-instruct",
            # "microsoft/phi-3-medium-4k-instruct",
            # "microsoft/phi-3-medium-128k-instruct",
            # "meta/llama3-70b-instruct",
            # "meta/llama3-8b-instruct",
            # "meta/llama2-70b",
            # "meta/codellama-70b",
            return [
                "stream",
                "temperature",
                "top_p",
                "frequency_penalty",
                "presence_penalty",
                "max_tokens",
                "max_completion_tokens",
                "stop",
                "seed",
                "tools",
                "tool_choice",
                "parallel_tool_calls",
                "response_format",
            ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_openai_params = self.get_supported_openai_params(model=model)
        for param, value in non_default_params.items():
            if param == "max_completion_tokens":
                optional_params["max_tokens"] = value
            elif param in supported_openai_params:
                optional_params[param] = value
        return optional_params

    def transform_response(  # type: ignore[override]
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: Any,
        request_data: dict,
        messages: List[Any],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        """
        Override transform_response to extract <think>…</think> reasoning blocks
        produced by NVIDIA NIM reasoning models (e.g. minimax/minimax-m1).

        NIM forwards the raw model output as a plain OpenAI-compatible response:
        the <think> block appears inside ``choices[0].message.content`` and
        ``reasoning_content`` is absent.  The parent class's
        ``_extract_reasoning_content`` helper already calls
        ``_parse_content_for_reasoning``, but only after the response has been
        deserialized.  We do the same pre-processing here on the raw JSON so
        that the upstream consumer always sees a clean split between
        ``reasoning_content`` and ``content``.
        """
        # Let the parent class build the base ModelResponse first.
        result = super().transform_response(
            model=model,
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=logging_obj,
            request_data=request_data,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            encoding=encoding,
            api_key=api_key,
            json_mode=json_mode,
        )

        # Post-process: if reasoning_content is still None but the content
        # contains <think>…</think> raw tags, extract them now.
        for choice in getattr(result, "choices", []):
            message = getattr(choice, "message", None)
            if message is None:
                continue
            if getattr(message, "reasoning_content", None) is not None:
                # Already parsed — nothing to do.
                continue
            content = getattr(message, "content", None)
            if not isinstance(content, str):
                continue
            reasoning, stripped_content = _parse_content_for_reasoning(content)
            if reasoning is not None:
                message.reasoning_content = reasoning
                message.content = stripped_content

        return result
