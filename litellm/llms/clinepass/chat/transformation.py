"""
Support for ClinePass (the Cline API) `/v1/chat/completions` endpoint.

ClinePass is OpenAI-compatible apart from two quirks, both handled here:

1. Non-streaming completions are nested under a ``data`` envelope --
   ``{"data": {"choices": [...]}, "success": true}`` -- rather than returning
   ``choices`` at the top level. Streaming responses are *not* wrapped, so the
   inherited SSE handling needs no change.
2. A bare model id is rejected with HTTP 400 ``invalid model format. Expected
   format: modelType/model``, but LiteLLM strips its own ``clinepass/`` routing
   prefix before the request is built, so a qualifier has to be restored.

Documentation: https://docs.cline.bot/

Credentials come only from the request's api_key or CLINEPASS_API_KEY.
Moderation and realtime endpoints are unsupported and rejected before dispatch.
"""

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Final, NoReturn

import httpx
from pydantic import TypeAdapter, ValidationError

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse

from ...openai.chat.gpt_transformation import OpenAIGPTConfig
from ..common_utils import ClinePassException

# Needed only for annotations; importing litellm_logging at runtime from a
# provider module risks a circular import.
if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.litellm_core_utils.tokenizer import Encoding as Tokenizer

CLINEPASS_API_BASE: Final = "https://api.cline.bot/api/v1"

# ClinePass nests the completion under this key on non-streaming responses.
CLINEPASS_RESPONSE_ENVELOPE_KEY: Final = "data"

# The qualifier ClinePass expects on outbound model ids.
#
# Note the hyphen: the catalog namespace is ``cline-pass/``, not ``clinepass/``
# (the latter is LiteLLM's own routing prefix, which is stripped before the
# request is built). The API only validates the *shape* of a model id -- any
# ``<segment>/<model>`` is accepted with HTTP 200 -- so an incorrect namespace
# fails silently rather than loudly. It is not inert, though: for at least one
# model an unrecognised namespace resolves to a different, date-pinned snapshot
# (``cline-pass/deepseek-v4-flash`` -> ``deepseek/deepseek-v4-flash``, while
# ``clinepass/deepseek-v4-flash`` -> ``deepseek/deepseek-v4-flash-0731``).
CLINEPASS_MODEL_PREFIX: Final = "cline-pass/"

# Headers that describe the original byte stream and would be wrong once the
# body is rewritten by _unwrap_response_envelope().
_BODY_SPECIFIC_HEADERS: Final = ("content-length", "content-encoding")

_JSON_OBJECT: Final = TypeAdapter(Mapping[str, object])


def _as_json_object(value: object) -> Mapping[str, object] | None:
    try:
        return _JSON_OBJECT.validate_python(value)
    except ValidationError:
        return None


def _unwrap_response_envelope(raw_response: httpx.Response) -> httpx.Response:
    """Strip ClinePass's ``data`` wrapper off a JSON completion body.

    The OpenAI transforms read ``raw_response.json()`` directly, so rebuild the
    response around the inner object rather than duplicating their bodies here.

    Returns the original response untouched whenever the body does not look like
    a wrapped completion, so an already-OpenAI-shaped body -- or an error nested
    under the same key -- is not mistaken for one.
    """
    try:
        payload: Final = _JSON_OBJECT.validate_json(raw_response.content)
    except ValidationError:
        # Not a JSON object body -- there is no envelope to strip.
        return raw_response

    if "choices" in payload:
        return raw_response

    inner: Final = _as_json_object(payload.get(CLINEPASS_RESPONSE_ENVELOPE_KEY))
    if inner is None or "choices" not in inner:
        return raw_response

    return httpx.Response(
        status_code=raw_response.status_code,
        headers={k: v for k, v in raw_response.headers.items() if k.lower() not in _BODY_SPECIFIC_HEADERS},
        content=json.dumps(inner, ensure_ascii=False).encode("utf-8"),
        request=_attached_request(raw_response),
    )


def _attached_request(raw_response: httpx.Response) -> httpx.Request | None:
    # httpx.Response.request raises RuntimeError rather than returning None when
    # no request is attached, so ask for it defensively instead of reaching for
    # the private attribute behind it.
    try:
        return raw_response.request
    except RuntimeError:
        return None


def _apply_model_prefix(data: dict[str, object]) -> dict[str, object]:  # mutable-ok: dict-typed base transform_request
    """Restore the ``modelType/model`` qualifier on the outbound model id.

    Only prefix ids that lost their qualifier, so a cross-provider id
    (``clinepass/openrouter/foo`` -> ``openrouter/foo``) is forwarded unchanged.
    """
    model: Final = data.get("model")
    if isinstance(model, str) and "/" not in model:
        return {**data, "model": f"{CLINEPASS_MODEL_PREFIX}{model}"}
    return data


class ClinePassConfig(OpenAIGPTConfig):
    """
    ClinePass configuration, inheriting the OpenAI chat transforms.

    Overrides only the request/response points where ClinePass diverges; see the
    module docstring for the two quirks.
    """

    @staticmethod
    def get_realtime_http_config(model: str) -> NoReturn:
        from litellm.exceptions import BadRequestError

        raise BadRequestError(
            message="ClinePass does not support realtime endpoints",
            model=model,
            llm_provider="clinepass",
        )

    @staticmethod
    def validate_moderation(model: str | None, custom_llm_provider: str | None = None) -> None:
        if custom_llm_provider != "clinepass" and not (model or "").startswith("clinepass/"):
            return
        from litellm.exceptions import BadRequestError

        raise BadRequestError(
            message="ClinePass does not support moderation endpoints",
            model=model or "",
            llm_provider="clinepass",
        )

    def _get_openai_compatible_provider_info(
        self, api_base: str | None, api_key: str | None
    ) -> tuple[str | None, str | None]:
        return (
            api_base or get_secret_str("CLINEPASS_API_BASE") or CLINEPASS_API_BASE,
            api_key or get_secret_str("CLINEPASS_API_KEY"),
        )

    def get_openai_compatible_provider_info(
        self,
        api_base: str | None,
        api_key: str | None,
    ) -> tuple[str | None, str | None]:
        return self._get_openai_compatible_provider_info(api_base, api_key)

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict[str, object],  # mutable-ok: matches the dict-typed base-class signature
        litellm_params: dict[str, object],  # mutable-ok: matches the dict-typed base-class signature
        stream: bool | None = None,
    ) -> str:
        base: Final = (api_base or CLINEPASS_API_BASE).rstrip("/")
        if base.endswith("/chat/completions"):
            return base

        return f"{base}/chat/completions"

    def get_models(
        self, api_key: str | None = None, api_base: str | None = None
    ) -> list[str]:  # mutable-ok: matches the dict-typed base-class signature
        """ClinePass exposes no model catalog.

        ``GET https://api.cline.bot/api/v1/models`` returns HTTP 404, and the
        inherited OpenAI implementation would additionally ask for it at the
        wrong path -- it rewrites the base URL down to scheme+host and appends
        ``/v1/models``. Return an empty catalog rather than making a request
        that is known to fail.
        """
        return []

    def map_openai_params(
        self,
        non_default_params: dict[str, object],  # mutable-ok: matches the dict-typed base-class signature
        optional_params: dict[str, object],  # mutable-ok: matches the dict-typed base-class signature
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:  # mutable-ok: matches the dict-typed base-class signature
        """ClinePass takes the legacy ``max_tokens`` spelling only."""
        mapped_params: Final[dict[str, object]]  # mutable-ok: dict-typed base map_openai_params
        mapped_params = super().map_openai_params(  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]  # base signature takes bare dict
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
        messages: list[AllMessageValues],  # mutable-ok: matches the dict-typed base-class signature
        optional_params: dict[str, object],  # mutable-ok: matches the dict-typed base-class signature
        litellm_params: dict[str, object],  # mutable-ok: matches the dict-typed base-class signature
        headers: dict[str, object],  # mutable-ok: matches the dict-typed base-class signature
    ) -> dict[str, object]:  # mutable-ok: matches the dict-typed base-class signature
        # BaseLLMHTTPHandler builds the body with this synchronous method on
        # both the sync and the async path, so there is deliberately no
        # async_transform_request() override -- it would never be called.
        data: Final[dict[str, object]]  # mutable-ok: dict-typed base transform_request
        data = super().transform_request(  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]  # base signature takes bare dict
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
        logging_obj: "Logging",
        request_data: dict[str, object],  # mutable-ok: matches the dict-typed base-class signature
        messages: list[AllMessageValues],  # mutable-ok: matches the dict-typed base-class signature
        optional_params: dict[str, object],  # mutable-ok: matches the dict-typed base-class signature
        litellm_params: dict[str, object],  # mutable-ok: matches the dict-typed base-class signature
        encoding: "Tokenizer | None",
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ModelResponse:
        # ClinePass was once observed returning finish_reason "stop" on a completion
        # cut off by max_tokens. Follow-up probes on 2026-08-22 did not reproduce it.
        # The provider therefore reports the upstream finish reason unmodified: inferring
        # truncation from usage equalling the cap produces false positives on natural
        # completions that happen to land exactly on the cap.
        return super().transform_response(  # pyright: ignore[reportUnknownMemberType]  # base signature takes bare dict
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
        self,
        error_message: str,
        status_code: int,
        headers: dict[str, object] | httpx.Headers,  # mutable-ok: matches the dict-typed base-class signature
    ) -> BaseLLMException:
        return ClinePassException(
            message=error_message,
            status_code=status_code,
            headers=headers,
        )
