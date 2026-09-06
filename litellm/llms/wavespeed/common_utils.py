"""
WaveSpeed AI common utilities.

WaveSpeed exposes every media model behind one asynchronous prediction API:

- ``POST {api_base}/api/v3/{model}`` submits a task and returns its id
- ``GET {api_base}/api/v3/predictions/{id}/result`` returns the task status and outputs

Both responses are wrapped in the platform envelope ``{"code": ..., "message": ..., "data": ...}``.

API Reference: https://wavespeed.ai/docs
"""

import base64
import mimetypes
import os
from collections.abc import Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import Final, Literal, TypedDict

import httpx
from typing_extensions import ReadOnly

from litellm._version import version as litellm_version
from litellm.litellm_core_utils.url_utils import encode_url_path_segment
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str


class WaveSpeedError(BaseLLMException):
    """Exception class for WaveSpeed AI API errors."""


DEFAULT_API_BASE: Final = "https://api.wavespeed.ai"
CHAT_API_BASE: Final = "https://llm.wavespeed.ai/v1"
DEFAULT_POLLING_INTERVAL: Final = 1.0
DEFAULT_MAX_POLLING_TIME: Final = 600
MAX_CONSECUTIVE_POLL_FAILURES: Final = 5

SUCCESS_STATUS: Final = "completed"
FAILURE_STATUSES: Final = frozenset({"failed", "cancelled", "timeout"})
PENDING_STATUSES: Final = frozenset({"created", "processing"})

OPENAI_STATUS_BY_WAVESPEED_STATUS: Final = MappingProxyType(
    {
        "created": "queued",
        "processing": "in_progress",
        "completed": "completed",
        "failed": "failed",
        "cancelled": "failed",
        "timeout": "failed",
    }
)


class WaveSpeedPrediction(TypedDict, total=False):
    id: ReadOnly[str]
    model: ReadOnly[str]
    status: ReadOnly[str]
    outputs: ReadOnly[Sequence[str]]
    error: ReadOnly[str]
    created_at: ReadOnly[str]
    has_nsfw_contents: ReadOnly[Sequence[bool]]


def to_request_payload(
    payload: Mapping[str, object] | Iterable[tuple[str, object]],
) -> dict:  # mutable-ok: base config contracts return bare `dict`
    """Materialize a read-only payload into the mutable ``dict`` the base config contracts declare."""
    return dict(payload)  # mutable-ok: base config contracts return bare `dict`


def optional_pair(key: str, value: object) -> tuple[tuple[str, object], ...]:
    """One key/value pair when the value is set, nothing otherwise, for splatting into a payload."""
    return ((key, value),) if value is not None else ()


def optional_entry(key: str, value: object) -> Mapping[str, object]:
    """One-entry mapping when the value is set, empty otherwise, for splatting into a payload."""
    return MappingProxyType({key: value}) if value is not None else MappingProxyType({})


_MAGIC_BYTE_MIME_TYPES: Final = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)


def _sniff_mime_type(payload: bytes) -> str:
    if payload[:4] == b"RIFF" and payload[8:12] == b"WEBP":
        return "image/webp"
    for magic, mime_type in _MAGIC_BYTE_MIME_TYPES:
        if payload.startswith(magic):
            return mime_type
    raise WaveSpeedError(
        status_code=400,
        message="Could not determine the media type of the reference. Pass a URL, a data URI, or a named file.",
    )


def _to_data_uri(payload: bytes, filename: str | None) -> str:
    guessed: Final = mimetypes.guess_type(filename)[0] if filename else None
    mime_type: Final = guessed or _sniff_mime_type(payload)
    return f"data:{mime_type};base64,{base64.b64encode(payload).decode()}"


def _reference_to_uri(reference: object, filename: str | None) -> str:
    if isinstance(reference, str):
        return reference
    if isinstance(reference, (bytes, bytearray)):
        return _to_data_uri(bytes(reference), filename)
    if isinstance(reference, os.PathLike):
        path: Final = os.fspath(reference)
        with open(path, "rb") as handle:
            return _to_data_uri(handle.read(), filename or str(path))
    read: Final = getattr(reference, "read", None)
    if callable(read):
        payload: Final = read()
        if not isinstance(payload, bytes):
            raise WaveSpeedError(status_code=400, message="Reference file handle must be opened in binary mode")
        return _to_data_uri(payload, filename or getattr(reference, "name", None))
    raise WaveSpeedError(status_code=400, message=f"Unsupported reference type: {type(reference).__name__}")


def to_reference_uri(reference: object) -> str:
    """Normalize an OpenAI ``input_reference`` into something a JSON body can carry.

    The shared video contract accepts URLs, raw bytes, paths, file handles and
    ``(filename, content)`` tuples, but WaveSpeed submits predictions as JSON, so
    anything that is not already a URL or data URI has to be inlined as one.
    """
    if not isinstance(reference, tuple):
        return _reference_to_uri(reference, None)
    if len(reference) < 2 or reference[1] is None:
        raise WaveSpeedError(status_code=400, message="Reference tuple is missing its content")
    supplied_name: Final = reference[0]
    return _reference_to_uri(reference[1], str(supplied_name) if supplied_name else None)


def get_api_key(api_key: str | None) -> str:
    resolved: Final = api_key or get_secret_str("WAVESPEED_API_KEY")
    if not resolved:
        raise WaveSpeedError(
            status_code=401,
            message="WaveSpeed API key is required. Set the WAVESPEED_API_KEY environment variable or pass api_key.",
        )
    return resolved


def get_api_base(api_base: str | None) -> str:
    """Resolve the base URL for the prediction API.

    Chat and media live on different hosts but share the ``wavespeed`` provider slug, so
    provider resolution and ``WAVESPEED_API_BASE`` can both hand this the chat base. That
    value would build an unreachable prediction URL, so it falls back to the media default.
    A self-hosted base is any other value and is honored as-is.
    """
    resolved: Final = (api_base or get_secret_str("WAVESPEED_API_BASE") or DEFAULT_API_BASE).rstrip("/")
    return DEFAULT_API_BASE if resolved == CHAT_API_BASE else resolved


def build_headers(api_key: str | None) -> Mapping[str, str]:
    """Auth plus the channel-attribution headers every WaveSpeed client sends."""
    return MappingProxyType(
        {
            "Authorization": f"Bearer {get_api_key(api_key)}",
            "Content-Type": "application/json",
            "X-Client-Name": "litellm",
            "X-Client-Version": litellm_version,
        }
    )


def build_submit_url(api_base: str | None, model: str) -> str:
    encoded_model: Final = "/".join(
        encode_url_path_segment(segment, field_name="model") for segment in model.split("/") if segment
    )
    if not encoded_model:
        raise WaveSpeedError(status_code=400, message="model is required for WaveSpeed predictions")
    return f"{get_api_base(api_base)}/api/v3/{encoded_model}"


def build_result_url(api_base: str | None, prediction_id: str) -> str:
    encoded_id: Final = encode_url_path_segment(prediction_id, field_name="prediction_id")
    return f"{get_api_base(api_base)}/api/v3/predictions/{encoded_id}/result"


def unwrap_envelope(raw_response: httpx.Response) -> WaveSpeedPrediction:
    """Return ``data`` from a WaveSpeed envelope, raising on transport or platform-level failure."""
    if raw_response.status_code >= 400:
        raise WaveSpeedError(
            status_code=raw_response.status_code,
            message=f"WaveSpeed request failed: {raw_response.text}",
            headers=raw_response.headers,
        )

    try:
        envelope: Final[object] = raw_response.json()
    except ValueError as e:
        raise WaveSpeedError(
            status_code=raw_response.status_code,
            message=f"Could not parse WaveSpeed response: {e}",
            headers=raw_response.headers,
        )

    if not isinstance(envelope, Mapping):
        raise WaveSpeedError(
            status_code=raw_response.status_code,
            message=f"Unexpected WaveSpeed response body: {raw_response.text}",
            headers=raw_response.headers,
        )

    code: Final = envelope.get("code")
    if code != 200:
        raise WaveSpeedError(
            status_code=raw_response.status_code,
            message=str(envelope.get("message") or f"WaveSpeed returned code {code}"),
            headers=raw_response.headers,
        )

    data: Final = envelope.get("data")
    if not isinstance(data, Mapping):
        raise WaveSpeedError(
            status_code=raw_response.status_code,
            message=f"WaveSpeed response is missing `data`: {raw_response.text}",
            headers=raw_response.headers,
        )
    return WaveSpeedPrediction(**data)


def get_prediction_id(prediction: WaveSpeedPrediction) -> str:
    prediction_id: Final = prediction.get("id")
    if not prediction_id:
        raise WaveSpeedError(status_code=500, message="WaveSpeed submit response is missing a prediction id")
    return prediction_id


def get_outputs(prediction: WaveSpeedPrediction) -> Sequence[str]:
    return prediction.get("outputs") or ()


def poll_outcome(prediction: WaveSpeedPrediction) -> Literal["done", "pending"]:
    """Classify a polled prediction, raising ``WaveSpeedError`` on a terminal failure."""
    status: Final = prediction.get("status", "")
    if status == SUCCESS_STATUS:
        return "done"
    if status in FAILURE_STATUSES:
        raise WaveSpeedError(
            status_code=400,
            message=f"WaveSpeed prediction {status}: {prediction.get('error') or 'no error detail returned'}",
        )
    return "pending"


def map_status_to_openai(status: str) -> str:
    return OPENAI_STATUS_BY_WAVESPEED_STATUS.get(status, "queued")
