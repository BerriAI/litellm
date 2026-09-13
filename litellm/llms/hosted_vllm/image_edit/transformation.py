from typing import Final

from litellm.llms.openai.image_edit.transformation import OpenAIImageEditConfig
from litellm.secret_managers.main import get_secret_str

PARAMS_VLLM_OMNI_DOES_NOT_ACCEPT: Final = frozenset({"mask", "quality", "input_fidelity"})


class HostedVLLMImageEditConfig(OpenAIImageEditConfig):
    def get_supported_openai_params(self, model: str) -> list:  # mutable-ok: BaseImageEditConfig contract
        return [  # mutable-ok: BaseImageEditConfig returns list
            param
            for param in super().get_supported_openai_params(model)
            if param not in PARAMS_VLLM_OMNI_DOES_NOT_ACCEPT
        ]

    def validate_environment(
        self,
        headers: dict,  # mutable-ok: BaseImageEditConfig contract
        model: str,
        api_key: str | None = None,
        litellm_params: dict | None = None,  # mutable-ok: BaseImageEditConfig contract
        api_base: str | None = None,
    ) -> dict:  # mutable-ok: BaseImageEditConfig contract
        resolved_key: Final = api_key or get_secret_str("HOSTED_VLLM_API_KEY") or "fake-api-key"
        return {**headers, "Authorization": f"Bearer {resolved_key}"}  # mutable-ok: httpx headers are a dict

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: dict,  # mutable-ok: BaseImageEditConfig contract
    ) -> str:
        resolved_api_base: Final = api_base or get_secret_str("HOSTED_VLLM_API_BASE")
        if resolved_api_base is None:
            raise ValueError(
                "api_base not set for Hosted VLLM images edits API. "
                "Set via api_base parameter or HOSTED_VLLM_API_BASE environment variable"
            )
        trimmed: Final = resolved_api_base.rstrip("/")
        if trimmed.endswith("/v1"):
            return f"{trimmed}/images/edits"
        return f"{trimmed}/v1/images/edits"
