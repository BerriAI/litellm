"""
Nadir Chat Completions API

Nadir (https://getnadir.com) is an intelligent LLM router. A single virtual
model, ``nadir/auto``, classifies each request by complexity and routes it to
the cheapest model that clears the quality bar (e.g. Haiku for simple prompts,
Sonnet for mid, Opus for complex), then returns an OpenAI-compatible response.

The endpoint speaks the OpenAI ``/v1/chat/completions`` dialect, so no request
translation is required. Nadir accepts the key as a Bearer token, so the
standard OpenAI-compatible transport works unchanged.

Cost attribution: the routed model name belongs to the underlying vendor, so it
does not resolve against a ``nadir/*`` pricing entry. Nadir returns the
authoritative cost it computed for the call, and ``transform_response`` below
surfaces it the same way the OpenRouter provider does. This is also why Nadir
has its own dispatch branch in ``main.py`` instead of riding the generic
OpenAI-compatible path, which never calls ``transform_response``.
"""

import httpx

from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse

# The OpenAI params Nadir's request schema actually accepts. Anything outside
# this set is dropped by the router rather than forwarded to the chosen model,
# so advertising more would be advertising a silent no-op. ``extra_headers``
# and ``max_retries`` are handled by the LiteLLM transport, not sent in the body.
_SUPPORTED_OPENAI_PARAMS = (
    "extra_headers",
    "frequency_penalty",
    "max_retries",
    "max_tokens",
    "presence_penalty",
    "response_format",
    "stream",
    "temperature",
    "top_p",
)


class NadirConfig(OpenAIGPTConfig):
    """
    Reference: https://getnadir.com/docs

    Nadir is OpenAI-compatible, so parameter mapping is inherited from
    ``OpenAIGPTConfig`` unchanged. ``model`` is a virtual router alias
    (``auto``); the concrete model is chosen server-side per request.
    """

    @classmethod
    def get_config(cls):
        return super().get_config()

    def get_supported_openai_params(self, model: str) -> list:  # mutable-ok: return type fixed by the base interface
        """
        Only the params Nadir's request schema actually accepts.

        Nadir speaks the OpenAI dialect but validates into its own request
        model, and anything outside that model is dropped rather than
        forwarded. Inheriting the full OpenAI param set would therefore
        advertise support that silently does nothing, so the list below is
        restricted to what the endpoint honors. ``extra_headers`` and
        ``max_retries`` are handled by the LiteLLM transport rather than sent
        in the body, so they stay.

        Notably absent: ``tools`` / ``tool_choice`` / ``functions``. Function
        calling is not part of the router's request schema today.
        """
        return list(_SUPPORTED_OPENAI_PARAMS)  # mutable-ok: the base interface returns a list

    def _get_openai_compatible_provider_info(self, api_base: "str | None", api_key: "str | None"):
        api_base = api_base or "https://api.getnadir.com/v1"
        return api_base, api_key

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: object,
        request_data: dict,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: object,
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ModelResponse:
        """
        Standard OpenAI response handling, plus Nadir's own cost.

        The routed model (``claude-haiku-4-5``, say) belongs to the underlying
        vendor, so it has no ``nadir/*`` pricing entry and the shared cost
        calculator cannot price it. Nadir already computes the cost of the call
        it actually made, so pass that through as the provider-reported cost
        rather than mirroring every vendor's price list under this provider.
        Same mechanism the OpenRouter provider uses.
        """
        model_response = super().transform_response(
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

        try:
            cost = raw_response.json()["nadir_metadata"]["cost"]["total_cost_usd"]
            if cost is not None:
                hidden = model_response._hidden_params
                if "additional_headers" not in hidden:
                    hidden["additional_headers"] = {}  # mutable-ok: the header bag the cost calculator reads
                hidden["additional_headers"]["llm_provider-x-litellm-response-cost"] = float(cost)
        except (ValueError, KeyError, TypeError):
            # Best-effort: a body that is not JSON, or is missing the cost
            # keys, is still a valid completion. Narrow rather than blind so a
            # genuine bug in here is not swallowed.
            pass

        return model_response
