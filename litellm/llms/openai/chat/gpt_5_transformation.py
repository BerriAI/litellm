"""Support for OpenAI gpt-5 model family."""

from typing import Optional, Union

import litellm
from litellm.utils import (
    _is_explicitly_disabled_factory,
    _supports_factory,
)

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


def _get_effort_level(value: Union[str, dict, None]) -> Optional[str]:
    """Extract the effective effort level from reasoning_effort (string or dict).

    Use this for guards that compare effort level (e.g. xhigh validation, "none" checks).
    Ensures dict inputs like {"effort": "none", "summary": "detailed"} are correctly
    treated as effort="none" for validation purposes.
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
        # The gpt-5-chat* family (gpt-5-chat, gpt-5-chat-latest, gpt-5-chat-2025-08-07,
        # …) are regular chat models: they support temperature and tool_choice but NOT
        # reasoning_effort.  They must NOT be routed through the GPT-5 reasoning path.
        #
        # Versioned chat models such as gpt-5.3-chat and gpt-5.1-chat ARE reasoning
        # models and must stay on the GPT-5 path.  The distinguishing feature is that
        # the gpt-5-chat family has a literal "-chat" immediately after "gpt-5"
        # (i.e. "gpt-5-chat…"), while versioned chat models interpose a minor version
        # number (i.e. "gpt-5.<digit>-chat").
        #
        # Using a startswith("gpt-5-chat") prefix check on the normalized name (rather
        # than a substring check) makes this boundary explicit and avoids any ambiguity
        # if future model names coincidentally contain "gpt-5-chat" as an interior run.
        _normalized = model.split("/")[-1]  # strip provider prefix, e.g. "openai/"
        return "gpt-5" in model and not _normalized.startswith("gpt-5-chat")

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
    def is_model_gpt_5_4_plus_model(cls, model: str) -> bool:
        """Check if the model is gpt-5.4 or newer (5.4, 5.5, 5.6, etc., including pro)."""
        model_name = model.split("/")[-1]
        if not model_name.startswith("gpt-5."):
            return False
        try:
            version_str = model_name.replace("gpt-5.", "").split("-")[0]
            major = version_str.split(".")[0]
            return int(major) >= 4
        except (ValueError, IndexError):
            return False

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

    @classmethod
    def _is_reasoning_effort_level_explicitly_disabled(
        cls, model: str, level: str
    ) -> bool:
        """Return True only when the model map explicitly sets the capability to False.

        Unlike ``_supports_reasoning_effort_level`` (which requires an explicit True),
        this method returns True only when ``supports_{level}_reasoning_effort`` is
        explicitly set to ``False`` in the model map.  A missing key is treated as
        supported (i.e. this method returns False = not disabled).

        Use this for opt-out checks where unknown models should be allowed through.
        """
        return _is_explicitly_disabled_factory(
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

        # Get raw reasoning_effort and effective effort level for all guards.
        # Use effective_effort (extracted string) for xhigh validation, "none" checks, and
        # tool/sampling guards — dict inputs like {"effort": "none", "summary": "detailed"}
        # must be treated as effort="none" to avoid incorrect tool-drop or sampling errors.
        raw_reasoning_effort = non_default_params.get(
            "reasoning_effort"
        ) or optional_params.get("reasoning_effort")
        effective_effort = _get_effort_level(raw_reasoning_effort)

        # Normalize dict reasoning_effort to string for Chat Completions API.
        # Example: {"effort": "high", "summary": "detailed"} -> "high"
        if isinstance(raw_reasoning_effort, dict) and "effort" in raw_reasoning_effort:
            normalized = _normalize_reasoning_effort_for_chat_completion(
                raw_reasoning_effort
            )
            if normalized is not None:
                if "reasoning_effort" in non_default_params:
                    non_default_params["reasoning_effort"] = normalized
                if "reasoning_effort" in optional_params:
                    optional_params["reasoning_effort"] = normalized

        if effective_effort == "xhigh":
            # xhigh is an opt-in capability: only allow if model explicitly supports it.
            if not self._supports_reasoning_effort_level(model, effective_effort):
                if litellm.drop_params or drop_params:
                    non_default_params.pop("reasoning_effort", None)
                    optional_params.pop("reasoning_effort", None)
                else:
                    raise litellm.utils.UnsupportedParamsError(
                        message=(
                            f"reasoning_effort={effective_effort} is not supported for this model."
                        ),
                        status_code=400,
                    )
        elif effective_effort in ("minimal", "low"):
            # minimal/low are opt-out: unknown models pass through; only block when
            # the model map explicitly sets supports_{level}_reasoning_effort=false.
            # Example: gpt-5.5-pro only accepts {medium, high, xhigh}, so it sets
            # supports_low_reasoning_effort=false (and supports_minimal=false).
            if self._is_reasoning_effort_level_explicitly_disabled(
                model, effective_effort
            ):
                if litellm.drop_params or drop_params:
                    non_default_params.pop("reasoning_effort", None)
                    optional_params.pop("reasoning_effort", None)
                else:
                    raise litellm.utils.UnsupportedParamsError(
                        message=(
                            f"reasoning_effort={effective_effort} is not supported for this model."
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

        # gpt-5.1/5.2 support logprobs, top_p, top_logprobs only when reasoning_effort="none"
        supports_none = self._supports_reasoning_effort_level(model, "none")
        if supports_none:
            sampling_params = ["logprobs", "top_logprobs", "top_p"]
            has_sampling = any(p in non_default_params for p in sampling_params)
            if has_sampling and effective_effort not in (None, "none"):
                if litellm.drop_params or drop_params:
                    for p in sampling_params:
                        non_default_params.pop(p, None)
                else:
                    raise litellm.utils.UnsupportedParamsError(
                        message=(
                            "gpt-5.1/5.2/5.4 only support logprobs, top_p, top_logprobs when "
                            "reasoning_effort='none'. Current reasoning_effort='{}'. "
                            "To drop unsupported params set `litellm.drop_params = True`"
                        ).format(effective_effort),
                        status_code=400,
                    )

        if "temperature" in non_default_params:
            temperature_value: Optional[float] = non_default_params.pop("temperature")
            if temperature_value is not None:
                # models supporting reasoning_effort="none" also support flexible temperature
                if supports_none and (
                    effective_effort == "none" or effective_effort is None
                ):
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
