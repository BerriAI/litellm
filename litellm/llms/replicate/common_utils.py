import asyncio
import time
from collections import deque
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import httpx
from pydantic import ConfigDict, TypeAdapter

import litellm
from litellm.constants import (
    DEFAULT_REPLICATE_POLLING_DELAY_SECONDS,
    DEFAULT_REPLICATE_SYNC_POLLING_RETRIES,
    REPLICATE_MAX_OUTPUT_DOWNLOAD_BYTES,
)
from litellm.litellm_core_utils.url_utils import (
    assert_same_origin,
    async_safe_get,
    encode_url_path_segment,
    ensure_api_route,
    join_api_base,
    safe_get,
)
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler, HTTPResponseLimitError
from litellm.secret_managers.main import get_secret_str

REPLICATE_DEFAULT_API_BASE: Final[str] = "https://api.replicate.com/v1"
REPLICATE_TERMINAL_STATUSES: Final[frozenset[str]] = frozenset({"succeeded", "failed", "canceled"})
_JSON_OBJECT: Final = TypeAdapter(Mapping[str, object], config=ConfigDict(hide_input_in_errors=True))
_REQUEST_HEADERS_NOT_FORWARDED: Final[frozenset[str]] = frozenset(
    {"host", "content-length", "content-type", "transfer-encoding", "prefer"}
)


class ReplicateError(BaseLLMException):
    def __init__(
        self,
        status_code: int,
        message: str,
        headers: dict[str, str] | httpx.Headers | None,
    ):
        super().__init__(status_code=status_code, message=message, headers=headers)


@dataclass(frozen=True, slots=True)
class ReplicatePredictionRequest:
    url: str
    body: Mapping[str, object]


def replicate_api_base(api_base: str | None) -> str:
    return ensure_api_route(api_base or get_secret_str("REPLICATE_API_BASE") or REPLICATE_DEFAULT_API_BASE, "v1")


def replicate_auth_headers(
    headers: Mapping[str, str], api_key: str | None
) -> dict[str, str]:  # mutable-ok: base class contracts return a plain dict
    resolved_key: Final = (
        api_key or litellm.replicate_key or get_secret_str("REPLICATE_API_KEY") or get_secret_str("REPLICATE_API_TOKEN")
    )
    if not resolved_key:
        raise ReplicateError(
            status_code=401,
            message="Replicate API key is required. Set REPLICATE_API_KEY or pass api_key.",
            headers=None,
        )
    return {**headers, "Authorization": f"Bearer {resolved_key}", "Content-Type": "application/json"}


def replicate_model_path(model: str) -> str:
    segments: Final = tuple(model.split("/"))
    if any(segment in ("", ".", "..") for segment in segments):
        raise ReplicateError(status_code=400, message=f"Invalid Replicate model path: {model!r}", headers={})
    return "/".join(encode_url_path_segment(segment, field_name="model") for segment in segments)


def replicate_prediction_request(
    api_base: str, model: str, prediction_input: Mapping[str, object]
) -> ReplicatePredictionRequest:
    if model.startswith("deployments/"):
        return ReplicatePredictionRequest(
            url=join_api_base(api_base, f"{replicate_model_path(model)}/predictions"),
            body=MappingProxyType({"input": prediction_input}),
        )
    if ":" in model:
        model_path, _, version = model.partition(":")
        if not version:
            raise ReplicateError(status_code=400, message=f"Invalid Replicate model version: {model!r}", headers={})
        return ReplicatePredictionRequest(
            url=join_api_base(
                api_base,
                f"models/{replicate_model_path(model_path)}/versions/"
                f"{encode_url_path_segment(version, field_name='version')}/predictions",
            ),
            body=MappingProxyType({"input": prediction_input}),
        )
    return ReplicatePredictionRequest(
        url=join_api_base(api_base, f"models/{replicate_model_path(model)}/predictions"),
        body=MappingProxyType({"input": prediction_input}),
    )


def replicate_prediction_url(api_base: str, prediction_id: str) -> str:
    return join_api_base(api_base, f"predictions/{encode_url_path_segment(prediction_id, field_name='prediction id')}")


def replicate_model_owner(model: str) -> str:
    return model.removeprefix("deployments/").split(":", 1)[0].split("/", 1)[0]


def replicate_prediction(raw_response: httpx.Response) -> Mapping[str, object]:
    try:
        return _JSON_OBJECT.validate_python(raw_response.json())
    except ValueError as error:
        raise ReplicateError(
            status_code=raw_response.status_code,
            message=f"Replicate returned a non-JSON response: {raw_response.text[:500]}",
            headers=raw_response.headers,
        ) from error


def replicate_prediction_string(prediction: Mapping[str, object], key: str) -> str | None:
    value: Final = prediction.get(key)
    return value if isinstance(value, str) and value else None


def replicate_prediction_input(prediction: Mapping[str, object]) -> Mapping[str, object]:
    raw_input: Final = prediction.get("input")
    return _JSON_OBJECT.validate_python(raw_input) if isinstance(raw_input, Mapping) else MappingProxyType({})


def replicate_prediction_metrics(prediction: Mapping[str, object]) -> Mapping[str, object]:
    raw_metrics: Final = prediction.get("metrics")
    return _JSON_OBJECT.validate_python(raw_metrics) if isinstance(raw_metrics, Mapping) else MappingProxyType({})


def replicate_positive_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _output_leaves(output: object) -> Iterator[object]:
    pending: Final = deque([output])
    while pending:
        if isinstance((item := pending.popleft()), Mapping):
            pending.extendleft(reversed(tuple(_JSON_OBJECT.validate_python(item).values())))
        elif isinstance(item, (list, tuple)):
            pending.extendleft(reversed(TypeAdapter(tuple[object, ...]).validate_python(item)))
        else:
            yield item


def replicate_output_urls(output: object) -> tuple[str, ...]:
    return tuple(
        leaf for leaf in _output_leaves(output) if isinstance(leaf, str) and leaf.startswith(("http://", "https://"))
    )


def replicate_prediction_failure(prediction: Mapping[str, object]) -> str | None:
    status: Final = replicate_prediction_string(prediction, "status")
    error: Final = replicate_prediction_string(prediction, "error")
    if status in ("failed", "canceled"):
        return error or f"Replicate prediction {status}"
    return None


def replicate_prediction_is_done(prediction: Mapping[str, object]) -> bool:
    return replicate_prediction_string(prediction, "status") in REPLICATE_TERMINAL_STATUSES


def replicate_poll_headers(
    request: httpx.Request,
) -> dict[str, str]:  # mutable-ok: HTTP handler clients take a plain dict
    return {key: value for key, value in request.headers.items() if key.lower() not in _REQUEST_HEADERS_NOT_FORWARDED}


def _poll_url(prediction: Mapping[str, object], raw_response: httpx.Response) -> str:
    urls: Final = prediction.get("urls")
    poll_url: Final = (
        replicate_prediction_string(_JSON_OBJECT.validate_python(urls), "get") if isinstance(urls, Mapping) else None
    )
    if poll_url is not None:
        assert_same_origin(poll_url, str(raw_response.request.url))
        return poll_url
    prediction_id: Final = replicate_prediction_string(prediction, "id")
    if prediction_id is None:
        raise ReplicateError(
            status_code=raw_response.status_code,
            message="Replicate prediction has no id to poll",
            headers=raw_response.headers,
        )
    api_base: Final = str(raw_response.request.url).split("/predictions", 1)[0].rsplit("/models/", 1)[0]
    return replicate_prediction_url(api_base.rsplit("/deployments/", 1)[0], prediction_id)


def _finished_prediction(raw_response: httpx.Response) -> Mapping[str, object] | None:
    if raw_response.status_code >= 400:
        raise ReplicateError(
            status_code=raw_response.status_code, message=raw_response.text, headers=raw_response.headers
        )
    prediction: Final = replicate_prediction(raw_response)
    return prediction if replicate_prediction_is_done(prediction) else None


def wait_for_replicate_prediction(
    raw_response: httpx.Response,
    client: HTTPHandler,
) -> Mapping[str, object]:
    prediction: Final = replicate_prediction(raw_response)
    if replicate_prediction_is_done(prediction):
        return prediction
    poll_url: Final = _poll_url(prediction, raw_response)
    headers: Final = replicate_poll_headers(raw_response.request)
    for _ in range(DEFAULT_REPLICATE_SYNC_POLLING_RETRIES):
        time.sleep(DEFAULT_REPLICATE_POLLING_DELAY_SECONDS)
        if (polled := _finished_prediction(client.get(url=poll_url, headers=headers))) is not None:  # pyright: ignore[reportUnknownMemberType]  # HTTP handler stubs are untyped
            return polled
    raise ReplicateError(
        status_code=504,
        message=f"Replicate prediction did not finish after {DEFAULT_REPLICATE_SYNC_POLLING_RETRIES} polls",
        headers=raw_response.headers,
    )


async def async_wait_for_replicate_prediction(
    raw_response: httpx.Response,
    client: AsyncHTTPHandler,
) -> Mapping[str, object]:
    prediction: Final = replicate_prediction(raw_response)
    if replicate_prediction_is_done(prediction):
        return prediction
    poll_url: Final = _poll_url(prediction, raw_response)
    headers: Final = replicate_poll_headers(raw_response.request)
    for _ in range(DEFAULT_REPLICATE_SYNC_POLLING_RETRIES):
        await asyncio.sleep(DEFAULT_REPLICATE_POLLING_DELAY_SECONDS)
        if (polled := _finished_prediction(await client.get(url=poll_url, headers=headers))) is not None:  # pyright: ignore[reportUnknownMemberType]  # HTTP handler stubs are untyped
            return polled
    raise ReplicateError(
        status_code=504,
        message=f"Replicate prediction did not finish after {DEFAULT_REPLICATE_SYNC_POLLING_RETRIES} polls",
        headers=raw_response.headers,
    )


def _download_too_large(url: str, error: HTTPResponseLimitError) -> ReplicateError:
    return ReplicateError(
        status_code=502,
        message=f"Replicate output at {url} exceeds REPLICATE_MAX_OUTPUT_DOWNLOAD_BYTES={REPLICATE_MAX_OUTPUT_DOWNLOAD_BYTES}: {error}",
        headers={},
    )


def replicate_download(client: HTTPHandler, url: str) -> httpx.Response:
    try:
        response: Final = safe_get(client, url, max_response_bytes=REPLICATE_MAX_OUTPUT_DOWNLOAD_BYTES)
    except HTTPResponseLimitError as error:
        raise _download_too_large(url, error) from error
    response.raise_for_status()
    return response


async def async_replicate_download(client: AsyncHTTPHandler, url: str) -> httpx.Response:
    try:
        response: Final = await async_safe_get(client, url, max_response_bytes=REPLICATE_MAX_OUTPUT_DOWNLOAD_BYTES)
    except HTTPResponseLimitError as error:
        raise _download_too_large(url, error) from error
    response.raise_for_status()
    return response
