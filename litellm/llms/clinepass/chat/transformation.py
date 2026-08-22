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
"""

import json
from typing import Any, List, Optional, Tuple, Union

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
CLINEPASS_MODEL_PREFIX = "cline-pass/"

# Headers that describe the original byte stream and would be wrong once the
# body is rewritten by _unwrap_response_envelope().
_BODY_SPECIFIC_HEADERS = ("content-length", "content-encoding")


def _as_positive_number(value: Any) -> Optional[float]:
    """Return ``value`` as a positive number, or ``None`` if it is not one.

    ``bool`` is rejected explicitly: it is a subclass of ``int``, and ``True``
    would otherwise read as a cap of 1.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value <= 0:
        return None
    return float(value)


def _correct_truncated_finish_reason(response: ModelResponse, request_data: dict) -> ModelResponse:
    """Report a truncated ClinePass completion as ``length``, not ``stop``.

    This is a defensive correction for an upstream bug in which ClinePass
    returned ``finish_reason: "stop"`` on a completion that had actually been
    cut off by ``max_tokens``: a request capped at 4000 came back with
    ``completion_tokens == 4000`` and still claimed a natural stop. Callers that
    trust ``finish_reason`` -- the documented way to detect truncation -- cannot
    then distinguish a complete answer from a guillotined one.

    Re-probing the live API later (2026-08-22, both streaming and non-streaming,
    caps of 2000 and 4000, across both the ``cline-pass/`` and the fallback
    namespace) did *not* reproduce the misreport: every capped response
    correctly returned ``length``. The upstream bug appears to have been fixed,
    or to be intermittent. This correction is therefore kept as a cheap safety
    net rather than as a workaround for a currently-observable defect, and it is
    deliberately conservative:

    - only an upstream ``stop`` is ever rewritten; ``length`` is already right,
    - only when usage shows the cap was actually reached,
    - and only for single-choice responses. ``usage.completion_tokens`` is an
      aggregate across all choices while ``max_tokens`` is a per-choice limit,
      so with ``n > 1`` the aggregate cannot identify *which* choice was
      truncated -- two naturally-finished 60-token choices under a cap of 100
      would otherwise both be relabelled ``length``.

    Streaming is deliberately not covered: chunks are assembled by the inherited
    SSE iterator, the terminal ``finish_reason`` arrives before the usage chunk
    that would justify rewriting it, and callers that omit
    ``stream_options.include_usage`` never receive usage at all. Since the
    misreport no longer reproduces, buffering the stream to correct it is not
    worth the latency and complexity.
    """
    if len(response.choices) != 1:
        return response

    max_tokens = _as_positive_number(request_data.get("max_tokens"))
    if max_tokens is None:
        max_tokens = _as_positive_number(request_data.get("max_completion_tokens"))
    if max_tokens is None:
        return response

    usage = getattr(response, "usage", None)
    completion_tokens = _as_positive_number(getattr(usage, "completion_tokens", None))
    if completion_tokens is None or completion_tokens < max_tokens:
        return response

    for choice in response.choices:
        if getattr(choice, "finish_reason", None) == "stop":
            choice.finish_reason = "length"

    return response


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

    # httpx.Response.request raises RuntimeError rather than returning None when
    # no request is attached, so ask for it defensively instead of reaching for
    # the private attribute behind it.
    try:
        original_request = raw_response.request
    except RuntimeError:
        original_request = None

    return httpx.Response(
        status_code=raw_response.status_code,
        headers=headers,
        content=json.dumps(inner).encode("utf-8"),
        request=original_request,
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

    def get_models(self, api_key: Optional[str] = None, api_base: Optional[str] = None) -> List[str]:
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
        # BaseLLMHTTPHandler builds the body with this synchronous method on
        # both the sync and the async path, so there is deliberately no
        # async_transform_request() override -- it would never be called.
        data = super().transform_request(
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
        response = super().transform_response(
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
        return _correct_truncated_finish_reason(response, request_data)

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return ClinePassException(
            message=error_message,
            status_code=status_code,
            headers=headers,
        )
