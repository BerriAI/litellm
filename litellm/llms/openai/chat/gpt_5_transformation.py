"""Support for OpenAI gpt-5 model family."""

from typing import Optional, Union

import litellm
from litellm.utils import _supports_factory

from .gpt_transformation import OpenAIGPTConfig


def _normalize_reasoning_effort_for_chat_completion(
    value: Union[str, dict, None],
) -> Optional[str]:
    """Convert reasoning_effort to the string format expected by OpenAI chat completion API.

    The chat completion API expects a simple string: 'none', 'low', 'medium', 'high', or 'xhigh'.
    Config/deployments may pass the Responses API format: {'effort': 'high', 'summary': 'detailed'}.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and "effort" in value:
        return value["effort"]
    return None


class OpenAIGPT5Config(OpenAIGPTConfig):
    """Configuration for gpt-5 models including GPT-5-Codex variants.

    Handles OpenAI API quirks for the gpt-5 series like:

    - Mapping ``max_tokens`` -> ``max_completion_tokens``.
    - Dropping unsupported ``temperature`` values when requested.
    - Support for GPT-5-Codex models optimized for code generation.
    """

    @classmethod
    def is_model_gpt_5_model(cls, model: str) -> bool:
        # gpt-5-chat* behaves like a regular chat model (supports temperature, etc.)
        # Don't route it through GPT-5 reasoning-specific parameter restrictions.
        return "gpt-5" in model and "gpt-5-chat" not in model

    @classmethod
    def is_model_gpt_5_search_model(cls, model: str) -> bool:
        """Check if the model is a GPT-5 search variant (e.g. gpt-5-search-api).

        Search-only models have a severely restricted parameter set compared to
        regular GPT-5 models.  They are identified by name convention (contain
        both ``gpt-5`` and ``search``).  Note: ``supports_web_search`` in model
        info is a *different* concept — it indicates a model can *use* web
        search as a tool, which many non-search-only models also support.
        """
        return "gpt-5" in model and "search" in model

    @classmethod
    def is_model_gpt_5_codex_model(cls, model: str) -> bool:
        """Check if the model is specifically a GPT-5 Codex variant."""
        return "gpt-5-codex" in model

    @classmethod
    def is_model_gpt_5_1_model(cls, model: str) -> bool:
        """Check if the model is a gpt-5.1 variant (e.g. gpt-5.1, gpt-5.1-codex)."""
        model_name = model.split("/")[-1]
        return model_name.startswith("gpt-5.1")

    @classmethod
    def is_model_gpt_5_2_model(cls, model: str) -> bool:
        """Check if the model is a gpt-5.2 variant (including pro)."""
        model_name = model.split("/")[-1]
        return model_name.startswith("gpt-5.2") or model_name.startswith("gpt-5.4")

    @classmethod
    def is_model_gpt_5_4_model(cls, model: str) -> bool:
        """Check if the model is a gpt-5.4 variant (including pro)."""
        model_name = model.split("/")[-1]
        return model_name.startswith("gpt-5.4")

    @classmethod
    def _supports_reasoning_effort_level(cls, model: str, level: str) -> bool:
        """Check if the model supports a specific reasoning_effort level.

        Looks up ``supports_{level}_reasoning_effort`` in the model map via
        the shared ``_supports_factory`` helper.
        Returns False for unknown models (safe fallback).
        """
        return _supports_factory(
            model=model,
            custom_llm_provider=None,
            key=f"supports_{level}_reasoning_effort",
        )

    def get_supported_openai_params(self, model: str) -> list:
        if self.is_model_gpt_5_search_model(model):
            return [
                "max_tokens",
                "max_completion_tokens",
                "stream",
                "stream_options",
                "web_search_options",
                "service_tier",
                "safety_identifier",
                "response_format",
                "user",
                "store",
                "verbosity",
                "max_retries",
                "extra_headers",
            ]

        from litellm.utils import supports_tool_choice

        base_gpt_series_params = super().get_supported_openai_params(model=model)
        gpt_5_only_params = ["reasoning_effort", "verbosity"]
        base_gpt_series_params.extend(gpt_5_only_params)
        if not supports_tool_choice(model=model):
            base_gpt_series_params.remove("tool_choice")

        non_supported_params = [
            "presence_penalty",
            "frequency_penalty",
            "stop",
            "logit_bias",
            "modalities",
            "prediction",
            "audio",
            "web_search_options",
        ]

        # gpt-5.1/5.2 support logprobs, top_p, top_logprobs when reasoning_effort="none"
        if not self._supports_reasoning_effort_level(model, "none"):
            non_supported_params.extend(["logprobs", "top_p", "top_logprobs"])

        return [
            param
            for param in base_gpt_series_params
            if param not in non_supported_params
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        if self.is_model_gpt_5_search_model(model):
            if "max_tokens" in non_default_params:
                optional_params["max_completion_tokens"] = non_default_params.pop(
                    "max_tokens"
                )
            return super()._map_openai_params(
                non_default_params=non_default_params,
                optional_params=optional_params,
                model=model,
                drop_params=drop_params,
            )

        # Normalize reasoning_effort: chat completion API expects a string, not a dict
        # (e.g. {'effort': 'high'} -> 'high')
        # BUT: preserve dict format if it has additional fields like 'summary' for Responses API
        raw_reasoning_effort = (
            non_default_params.get("reasoning_effort")
            or optional_params.get("reasoning_effort")
        )
        
        # Only normalize if it's a simple dict with just 'effort' key
        # Preserve dict format if it has additional fields (e.g., 'summary') for Responses API
        should_normalize = False
        if isinstance(raw_reasoning_effort, dict):
            # Only normalize if dict has only 'effort' key (or is empty)
            if set(raw_reasoning_effort.keys()) == {"effort"} or len(raw_reasoning_effort) == 0:
                should_normalize = True
        elif isinstance(raw_reasoning_effort, str):
            # String format is already normalized
            should_normalize = False
        
        if should_normalize:
            normalized = _normalize_reasoning_effort_for_chat_completion(raw_reasoning_effort)
            if raw_reasoning_effort is not None and normalized is not None:
                if "reasoning_effort" in non_default_params:
                    non_default_params["reasoning_effort"] = normalized
                if "reasoning_effort" in optional_params:
                    optional_params["reasoning_effort"] = normalized
        else:
            # Keep the original format (string or dict with additional fields)
            normalized = raw_reasoning_effort

        reasoning_effort = normalized or raw_reasoning_effort
        if reasoning_effort is not None and reasoning_effort == "xhigh":
            if not self._supports_reasoning_effort_level(model, "xhigh"):
                if litellm.drop_params or drop_params:
                    non_default_params.pop("reasoning_effort", None)
                else:
                    raise litellm.utils.UnsupportedParamsError(
                        message=(
                            "reasoning_effort='xhigh' is only supported for gpt-5.1-codex-max, gpt-5.2, and gpt-5.4+ models."
                        ),
                        status_code=400,
                    )

        ################################################################
        # max_tokens is not supported for gpt-5 models on OpenAI API
        # Relevant issue: https://github.com/BerriAI/litellm/issues/13381
        ################################################################
        if "max_tokens" in non_default_params:
            optional_params["max_completion_tokens"] = non_default_params.pop(
                "max_tokens"
            )

        # gpt-5.4: function calls not supported when reasoning_effort != "none" in chat completions API
        # However, the Responses API supports both tools and reasoning together
        # So we keep reasoning_effort if the request will be routed to Responses API
        if self.is_model_gpt_5_4_model(model):
            has_tools = bool(
                non_default_params.get("tools") or optional_params.get("tools")
            )
            if has_tools and reasoning_effort not in (None, "none"):
                # Check if this will be routed to Responses API
                # If so, keep reasoning_effort; otherwise drop it for chat completions API
                model_name = model.split("/")[-1]
                will_route_to_responses = False
                if model_name.startswith("gpt-5."):
                    try:
                        version_str = model_name.replace("gpt-5.", "").split("-")[0]
                        if "." in version_str:
                            major_version = int(version_str.split(".")[0])
                        else:
                            major_version = int(version_str)
                        will_route_to_responses = major_version >= 4
                    except (ValueError, IndexError):
                        pass
                
                if not will_route_to_responses:
                    non_default_params.pop("reasoning_effort", None)
                    optional_params.pop("reasoning_effort", None)
                    reasoning_effort = None

        # gpt-5.1/5.2 support logprobs, top_p, top_logprobs only when reasoning_effort="none"
        supports_none = self._supports_reasoning_effort_level(model, "none")
        if supports_none:
            sampling_params = ["logprobs", "top_logprobs", "top_p"]
            has_sampling = any(p in non_default_params for p in sampling_params)
            if has_sampling and reasoning_effort not in (None, "none"):
                if litellm.drop_params or drop_params:
                    for p in sampling_params:
                        non_default_params.pop(p, None)
                else:
                    raise litellm.utils.UnsupportedParamsError(
                        message=(
                            "gpt-5.1/5.2/5.4 only support logprobs, top_p, top_logprobs when "
                            "reasoning_effort='none'. Current reasoning_effort='{}'. "
                            "To drop unsupported params set `litellm.drop_params = True`"
                        ).format(reasoning_effort),
                        status_code=400,
                    )

        if "temperature" in non_default_params:
            temperature_value: Optional[float] = non_default_params.pop("temperature")
            if temperature_value is not None:
                # models supporting reasoning_effort="none" also support flexible temperature
                if supports_none and (reasoning_effort == "none" or reasoning_effort is None):
                    optional_params["temperature"] = temperature_value
                elif temperature_value == 1:
                    optional_params["temperature"] = temperature_value
                elif litellm.drop_params or drop_params:
                    pass
                else:
                    raise litellm.utils.UnsupportedParamsError(
                        message=(
                            "gpt-5 models (including gpt-5-codex) don't support temperature={}. "
                            "Only temperature=1 is supported. "
                            "For gpt-5.1, temperature is supported when reasoning_effort='none' (or not specified, as it defaults to 'none'). "
                            "To drop unsupported params set `litellm.drop_params = True`"
                        ).format(temperature_value),
                        status_code=400,
                    )
        return super()._map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=drop_params,
        )
