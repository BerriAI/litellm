"""
Translates from OpenAI's `/v1/chat/completions` to Auxen's `/v1/chat/completions`.

Auxen (https://auxen.ai) hosts per-customer dedicated LLM endpoints. Each
provisioned instance exposes an OpenAI-compatible HTTP API at
    https://api.auxen.ai/v1/<instance_id>/v1/chat/completions
and is authenticated with a per-instance `auxk_*` bearer token.

The provider is intentionally a thin pass-through over OpenAI's request/response
schema — no parameter remapping is needed. Users set:

    AUXEN_API_BASE  → e.g. https://api.auxen.ai/v1/inst_xxx/v1
    AUXEN_API_KEY   → the auxk_* key issued by their Auxen dashboard

then call models as `auxen/<model-name>` (the model name is informational; the
instance is already locked to one model at provision time).
"""

from typing import Optional, Tuple

from litellm.secret_managers.main import get_secret_str

from ...openai.chat.gpt_transformation import OpenAIGPTConfig


class AuxenChatConfig(OpenAIGPTConfig):
    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        api_base = api_base or get_secret_str("AUXEN_API_BASE")
        dynamic_api_key = api_key or get_secret_str("AUXEN_API_KEY")
        return api_base, dynamic_api_key

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        if not api_base:
            raise ValueError(
                "Auxen requires an `api_base` (set AUXEN_API_BASE or pass "
                "`api_base=...`). Per-instance base URLs are issued by the "
                "Auxen dashboard and look like "
                "https://api.auxen.ai/v1/inst_xxx/v1"
            )

        if not api_base.endswith("/chat/completions"):
            api_base = f"{api_base.rstrip('/')}/chat/completions"

        return api_base
