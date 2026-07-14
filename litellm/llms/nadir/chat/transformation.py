"""
Nadir Chat Completions API

Nadir (https://getnadir.com) is an intelligent LLM router. A single virtual
model, ``nadir/auto``, classifies each request by complexity and routes it to
the cheapest model that clears the quality bar (e.g. Haiku for simple prompts,
Sonnet for mid, Opus for complex), then returns an OpenAI-compatible response.

The endpoint speaks the OpenAI ``/v1/chat/completions`` dialect, so no request
translation is required. The response reports ``model`` as the model that was
actually routed to (not ``auto``), which lets LiteLLM's cost tracking price the
real underlying model. Nadir accepts the key as a Bearer token, so the standard
OpenAI-compatible transport works unchanged.
"""

from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig


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

    def get_supported_openai_params(self, model: str) -> list:
        """
        Nadir forwards standard OpenAI chat params to whichever model it routes
        to, so the full OpenAI-compatible param set is supported.
        """
        return super().get_supported_openai_params(model=model)

    def _get_openai_compatible_provider_info(self, api_base: "str | None", api_key: "str | None"):
        api_base = api_base or "https://api.getnadir.com/v1"
        return api_base, api_key
