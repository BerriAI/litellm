"""
GLM (ZhipuAI) Chat Completion Transformation

GLM's API is OpenAI-compatible (same message format, streaming SSE, and tool calls),
so this config simply extends OpenAIGPTConfig and only overrides URL construction,
authentication (Bearer token injection), and sampling-parameter mapping.

Sampling parameter notes (from GLM API docs):
  - temperature : float, 0–1   (higher → more random)
  - top_p       : float, 0.01–1
  - max_tokens  : int,   1–131072
  - do_sample   : bool   — LiteLLM-side flag, NOT forwarded to the API.
                  When False, temperature and top_p are stripped from the
                  request so that the model uses greedy decoding.

API reference: https://open.bigmodel.cn/dev/api/normal-model/glm-4
"""

import os
from typing import List, Optional, Union

import httpx

import litellm
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.types.llms.openai import AllMessageValues


GLM_API_BASE = "https://open.bigmodel.cn/api/paas/v4"

# GLM hard limits on sampling parameters
GLM_MAX_TOKENS = 131_072
GLM_TEMPERATURE_MIN = 0.0
GLM_TEMPERATURE_MAX = 1.0
GLM_TOP_P_MIN = 0.01
GLM_TOP_P_MAX = 1.0


class GLMError(BaseLLMException):
    """GLM-specific exception, wrapping HTTP errors from the GLM API."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(status_code=status_code, message=message)


class GLMChatConfig(OpenAIGPTConfig):
    """
    Chat completion configuration for ZhipuAI GLM provider.

    Inherits all OpenAI-compatible behaviour from OpenAIGPTConfig and overrides:
    - API base URL resolution
    - Bearer-token authentication header injection
    - Sampling-parameter mapping (do_sample, temperature, top_p, max_tokens)
    - Error class
    """

    # -----------------------------------------------------------------
    # URL helpers
    # -----------------------------------------------------------------

    def _get_openai_compatible_provider_info(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
    ):
        """Return (resolved_api_base, resolved_api_key)."""
        resolved_base = api_base or GLM_API_BASE
        resolved_key = (
            api_key
            or litellm.api_key
            or os.environ.get("GLM_API_KEY")
        )
        return resolved_base, resolved_key

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict = {},
        stream: Optional[bool] = None,
    ) -> str:
        """Build the full endpoint URL, avoiding double-suffixing."""
        base, _ = self._get_openai_compatible_provider_info(api_base, api_key)
        base = base.rstrip("/")
        if not base.endswith("/chat/completions"):
            base = f"{base}/chat/completions"
        return base

    # -----------------------------------------------------------------
    # Authentication
    # -----------------------------------------------------------------

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        """Inject Bearer auth header and content-type headers."""
        _, resolved_key = self._get_openai_compatible_provider_info(api_base, api_key)

        default_headers = {
            "accept": "application/json",
            "content-type": "application/json",
        }
        if resolved_key:
            default_headers["Authorization"] = f"Bearer {resolved_key}"

        # Caller-supplied headers take precedence
        return {**default_headers, **headers}

    # -----------------------------------------------------------------
    # Sampling parameters
    # -----------------------------------------------------------------

    def get_supported_openai_params(self, model: str) -> list:
        """
        Return the list of params understood by this provider.

        Includes ``do_sample`` as a LiteLLM-side control flag; it is
        consumed in ``map_openai_params`` and never forwarded to the API.
        """
        parent_params = super().get_supported_openai_params(model=model)
        # do_sample is a GLM/HuggingFace convention; we handle it ourselves
        return parent_params + ["do_sample"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Map OpenAI-style params to the GLM API payload.

        Special behaviour:
        - ``do_sample=False``  → remove temperature and top_p (greedy decoding)
        - ``do_sample=True``   → no-op; temperature/top_p pass through normally
        - ``do_sample`` itself is never forwarded to the API
        - ``max_tokens``       → clamped to GLM_MAX_TOKENS (131 072)
        - ``temperature``      → clamped to [0.0, 1.0]
        - ``top_p``            → clamped to [0.01, 1.0]
        """
        # First, let the parent map all standard OpenAI params.
        # Because we added "do_sample" to get_supported_openai_params, the
        # parent's _map_openai_params loop will copy it into optional_params
        # if the caller passed it.
        optional_params = super().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=drop_params,
        )

        # Consume do_sample — it must never be sent to the GLM API.
        do_sample: Optional[bool] = optional_params.pop("do_sample", None)

        # When do_sample is explicitly False, use greedy decoding by
        # stripping temperature and top_p from the request.
        if do_sample is False:
            optional_params.pop("temperature", None)
            optional_params.pop("top_p", None)

        # Clamp max_tokens to GLM's hard limit.
        if "max_tokens" in optional_params:
            optional_params["max_tokens"] = min(
                int(optional_params["max_tokens"]), GLM_MAX_TOKENS
            )

        # Clamp temperature to [0.0, 1.0].
        if "temperature" in optional_params:
            optional_params["temperature"] = max(
                GLM_TEMPERATURE_MIN,
                min(float(optional_params["temperature"]), GLM_TEMPERATURE_MAX),
            )

        # Clamp top_p to [0.01, 1.0].
        if "top_p" in optional_params:
            optional_params["top_p"] = max(
                GLM_TOP_P_MIN,
                min(float(optional_params["top_p"]), GLM_TOP_P_MAX),
            )

        return optional_params

    # -----------------------------------------------------------------
    # Error handling
    # -----------------------------------------------------------------

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return GLMError(status_code=status_code, message=error_message)
