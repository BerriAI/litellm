"""
Nvidia NIM endpoint: https://docs.api.nvidia.com/nim/reference/databricks-dbrx-instruct-infer

This is OpenAI compatible

This file only contains param mapping logic

API calling is done using the OpenAI SDK with an api_base
"""

from collections.abc import Collection, Mapping
from typing import Final, cast  # noqa: TID251  # litellm.model_cost is an untyped module global

import litellm
from litellm.exceptions import UnsupportedParamsError
from litellm.litellm_core_utils.core_helpers import as_str_mapping, drops_unsupported_params
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.llms.openai_like.dynamic_config import MapsAllowedOpenAIParams


def _serves_logprobs(model: str) -> bool:
    catalog: Final = cast(Mapping[str, object], litellm.model_cost)  # cast-ok: validating the map per call is slow
    row: Final = as_str_mapping(catalog.get(f"nvidia_nim/{model}"))
    return row is None or row.get("supports_logprobs") is not False


def _malformed_logprobs_params(non_default_params: Mapping[str, object]) -> frozenset[str]:
    logprobs: Final = non_default_params.get("logprobs")
    non_boolean_logprobs: Final = "logprobs" in non_default_params and not isinstance(logprobs, bool)
    top_logprobs_without_logprobs: Final = "top_logprobs" in non_default_params and logprobs is not True
    return frozenset(
        param
        for param, malformed in (("logprobs", non_boolean_logprobs), ("top_logprobs", top_logprobs_without_logprobs))
        if malformed
    )


class NvidiaNimConfig(OpenAIGPTConfig, MapsAllowedOpenAIParams):
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
                *(("logprobs", "top_logprobs") if _serves_logprobs(model) else ()),
            ]

    def map_openai_params(
        self,
        non_default_params: dict[str, object],  # mutable-ok: get_optional_params passes its own dicts
        optional_params: dict[str, object],  # mutable-ok: filled in place for get_optional_params
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:  # mutable-ok: get_optional_params keeps building on the mapped params
        return self.map_openai_params_allowing(non_default_params, optional_params, model, drop_params, ())

    def map_openai_params_allowing(
        self,
        non_default_params: dict[str, object],  # mutable-ok: same caller dicts map_openai_params takes
        optional_params: dict[str, object],  # mutable-ok: filled in place like map_openai_params
        model: str,
        drop_params: bool,
        allowed_openai_params: Collection[str],
    ) -> dict[str, object]:  # mutable-ok: get_optional_params keeps building on the mapped params
        supported_openai_params: Final = self.get_supported_openai_params(model=model)
        malformed_params: Final = _malformed_logprobs_params(non_default_params) - frozenset(allowed_openai_params)
        if malformed_params and not drops_unsupported_params(drop_params):
            raise UnsupportedParamsError(
                message=(
                    f"nvidia_nim does not support parameters: {sorted(malformed_params)}, for model={model}. "
                    "NVIDIA takes logprobs as true or false, and returns top_logprobs only with logprobs=true. "
                    "To drop these, set `litellm.drop_params=True`, or send "
                    f"allowed_openai_params={sorted(malformed_params)} to pass them as is."
                ),
                llm_provider="nvidia_nim",
                model=model,
            )
        for param, value in non_default_params.items():
            if param == "max_completion_tokens":
                optional_params["max_tokens"] = value
            elif param in supported_openai_params and param not in malformed_params:
                optional_params[param] = value
        return optional_params
