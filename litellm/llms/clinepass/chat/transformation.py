"""
Support for ClinePass (the Cline API) `/v1/chat/completions` endpoint.

ClinePass is OpenAI-compatible apart from two quirks, both handled here:

1. Non-streaming completions are nested under a ``data`` envelope --
   ``{"data": {"choices": [...]}, "success": true}`` -- rather than returning
   ``choices`` at the top level. Streaming responses are *not* wrapped, so the
   inherited SSE handling needs no change.
2. A bare model id is rejected with HTTP 400 ``invalid model format. Expected
   format: modelType/model``, but LiteLLM strips its own ``clinepass/`` routing
   prefix before the request is built, so it has to be restored.

Documentation: https://docs.cline.bot/
"""

import json
from typing import Any, List, Tuple, Union

import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse

from ...openai.chat.gpt_transformation import OpenAIGPTConfig
from ..common_utils import ClinePassException

CLINEPASS_API_BASE = "https://api.cline.bot/api/v1"

# ClinePass nests the completion under this key on non-streaming responses.
CLINEPASS_RESPONSE_ENVELOPE_KEY = "data"

# The qualifier ClinePass requires on outbound model ids.
CLINEPASS_MODEL_PREFIX = "clinepass/"

# Headers that describe the original byte stream and would be wrong once the
# body is rewritten by _unwrap_response_envelope().
_BODY_SPECIFIC_HEADERS = ("content-length", "content-encoding")


def _unwrap_response_envelope(raw_response: httpx.Response) -> httpx.Response:
    """Strip ClinePass's ``data`` wrapper off a JSON completion body.

    The OpenAI transforms read ``raw_response.json()`` directly, so rebuild the
    response around the inner object rather than duplicating their bodies here.

    Returns the original response untouched whenever the body does not look like
    a wrapped completion, so an already-OpenAI-shaped body -- or an error nested
    under the same key -- is not mistaken for one.
    """
    try:
        payload = raw_response.json()
    except (ValueError, httpx.StreamError):
        # Not a JSON body, or a streaming response that has not been read --
        # either way there is no envelope to strip.
        return raw_response

    if not isinstance(payload, dict) or "choices" in payload:
        return raw_response

    inner = payload.get(CLINEPASS_RESPONSE_ENVELOPE_KEY)
    if not isinstance(inner, dict) or "choices" not in inner:
        return raw_response

    headers = {k: v for k, v in raw_response.headers.items() if k.lower() not in _BODY_SPECIFIC_HEADERS}

    return httpx.Response(
        status_code=raw_response.status_code,
        headers=headers,
        content=json.dumps(inner).encode("utf-8"),
        request=getattr(raw_response, "_request", None),
    )


def _apply_model_prefix(data: dict) -> dict:
    """Restore the ``modelType/model`` qualifier on the outbound model id.

    Only prefix ids that lost their qualifier, so a cross-provider id
    (``clinepass/openrouter/foo`` -> ``openrouter/foo``) is forwarded unchanged.
    """
    model = data.get("model")
    if isinstance(model, str) and "/" not in model:
        data["model"] = f"{CLINEPASS_MODEL_PREFIX}{model}"
    return data


class ClinePassConfig(OpenAIGPTConfig):
    """
    ClinePass configuration, inheriting the OpenAI chat transforms.

    Overrides only the request/response points where ClinePass diverges; see the
    module docstring for the two quirks.
    """

    def _get_openai_compatible_provider_info(
        self, api_base: str | None, api_key: str | None
    ) -> Tuple[str | None, str | None]:
        api_base = api_base or get_secret_str("CLINEPASS_API_BASE") or CLINEPASS_API_BASE
        dynamic_api_key = api_key or get_secret_str("CLINEPASS_API_KEY")
        return api_base, dynamic_api_key

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        if not api_base:
            api_base = CLINEPASS_API_BASE

        api_base = api_base.rstrip("/")
        if api_base.endswith("/chat/completions"):
            return api_base

        return f"{api_base}/chat/completions"

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """ClinePass takes the legacy ``max_tokens`` spelling only."""
        mapped_params = super().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=drop_params,
        )
        if "max_completion_tokens" in mapped_params:
            mapped_params["max_tokens"] = mapped_params.pop("max_completion_tokens")
        return mapped_params

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        data = super().transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )
        return _apply_model_prefix(data)

    async def async_transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        data = await super().async_transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )
        return _apply_model_prefix(data)

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: Any,
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ModelResponse:
        return super().transform_response(
            model=model,
            raw_response=_unwrap_response_envelope(raw_response),
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

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return ClinePassException(
            message=error_message,
            status_code=status_code,
            headers=headers,
        )
