"""
WaveSpeed AI image generation handler.

WaveSpeed predictions are asynchronous: one submit POST returns a prediction id, then the
result endpoint is polled until the prediction reaches a terminal status.

The submit POST is issued exactly once and is never retried, because every submission is a
billable task and a retry would create a duplicate one. Poll GETs are read-only, so a short
run of connection failures is tolerated before giving up.
"""

import asyncio
import time
from collections.abc import Coroutine, Mapping
from types import MappingProxyType
from typing import Final, NamedTuple

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,  # pyright: ignore[reportPrivateUsage]  # the shared sync client factory litellm providers use
    get_async_httpx_client,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import ImageResponse

from ..common_utils import (
    DEFAULT_MAX_POLLING_TIME,
    DEFAULT_POLLING_INTERVAL,
    MAX_CONSECUTIVE_POLL_FAILURES,
    WaveSpeedError,
    build_result_url,
    get_prediction_id,
    poll_outcome,
    to_request_payload,
    unwrap_envelope,
)
from .transformation import WaveSpeedImageGenerationConfig


class _PreparedRequest(NamedTuple):
    headers: Mapping[str, str]
    submit_url: str
    body: Mapping[str, object]


class _ResolvedParams(NamedTuple):
    api_key: str | None
    api_base: str | None
    litellm_params: Mapping[str, object]


class WaveSpeedImageGeneration:
    def __init__(self, config: WaveSpeedImageGenerationConfig | None = None) -> None:
        self.config: Final = config or WaveSpeedImageGenerationConfig()

    def image_generation(
        self,
        model: str,
        prompt: str,
        model_response: ImageResponse,
        optional_params: Mapping[str, object],
        litellm_params: GenericLiteLLMParams | Mapping[str, object],
        logging_obj: LiteLLMLoggingObj,
        timeout: float | httpx.Timeout | None,
        extra_headers: Mapping[str, str] | None = None,
        client: HTTPHandler | AsyncHTTPHandler | None = None,
        aimg_generation: bool = False,
    ) -> "ImageResponse | Coroutine[object, object, ImageResponse]":
        if aimg_generation:
            return self.async_image_generation(
                model=model,
                prompt=prompt,
                model_response=model_response,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logging_obj=logging_obj,
                timeout=timeout,
                extra_headers=extra_headers,
                client=client if isinstance(client, AsyncHTTPHandler) else None,
            )

        resolved: Final = _resolve_params(litellm_params)
        sync_client: Final = client if isinstance(client, HTTPHandler) else _get_httpx_client()
        prepared: Final = self._prepare(model, prompt, resolved, optional_params, extra_headers, logging_obj)

        submit_response: Final = sync_client.post(
            url=prepared.submit_url,
            headers=to_request_payload(prepared.headers),
            json=to_request_payload(prepared.body),
            timeout=timeout,
        )
        prediction_id: Final = get_prediction_id(unwrap_envelope(submit_response))
        result_url: Final = build_result_url(resolved.api_base, prediction_id)
        deadline: Final = time.time() + DEFAULT_MAX_POLLING_TIME
        poll_headers: Final = to_request_payload(prepared.headers)

        consecutive_failures = 0  # rebind-ok: counts consecutive poll transport failures
        while time.time() < deadline:
            try:
                poll_response = sync_client.get(url=result_url, headers=poll_headers, timeout=timeout)
            except Exception as e:  # noqa: BLE001  # any transport failure is retried, never the billable submit
                consecutive_failures = _record_poll_failure(consecutive_failures, prediction_id, e)
                time.sleep(DEFAULT_POLLING_INTERVAL)
                continue

            consecutive_failures = 0
            if poll_outcome(unwrap_envelope(poll_response)) == "done":
                return self._transform(
                    model, poll_response, model_response, logging_obj, prepared, optional_params, resolved
                )

            time.sleep(DEFAULT_POLLING_INTERVAL)

        raise _timeout_error(prediction_id)

    async def async_image_generation(
        self,
        model: str,
        prompt: str,
        model_response: ImageResponse,
        optional_params: Mapping[str, object],
        litellm_params: GenericLiteLLMParams | Mapping[str, object],
        logging_obj: LiteLLMLoggingObj,
        timeout: float | httpx.Timeout | None,
        extra_headers: Mapping[str, str] | None = None,
        client: AsyncHTTPHandler | None = None,
    ) -> ImageResponse:
        resolved: Final = _resolve_params(litellm_params)
        async_client: Final = client or get_async_httpx_client(llm_provider=litellm.LlmProviders.WAVESPEED)
        prepared: Final = self._prepare(model, prompt, resolved, optional_params, extra_headers, logging_obj)

        submit_response: Final = await async_client.post(
            url=prepared.submit_url,
            headers=to_request_payload(prepared.headers),
            json=to_request_payload(prepared.body),
            timeout=timeout,
        )
        prediction_id: Final = get_prediction_id(unwrap_envelope(submit_response))
        result_url: Final = build_result_url(resolved.api_base, prediction_id)
        deadline: Final = time.time() + DEFAULT_MAX_POLLING_TIME
        poll_headers: Final = to_request_payload(prepared.headers)

        consecutive_failures = 0  # rebind-ok: counts consecutive poll transport failures
        while time.time() < deadline:
            try:
                poll_response = await async_client.get(url=result_url, headers=poll_headers, timeout=timeout)
            except Exception as e:  # noqa: BLE001  # any transport failure is retried, never the billable submit
                consecutive_failures = _record_poll_failure(consecutive_failures, prediction_id, e)
                await asyncio.sleep(DEFAULT_POLLING_INTERVAL)
                continue

            consecutive_failures = 0
            if poll_outcome(unwrap_envelope(poll_response)) == "done":
                return self._transform(
                    model, poll_response, model_response, logging_obj, prepared, optional_params, resolved
                )

            await asyncio.sleep(DEFAULT_POLLING_INTERVAL)

        raise _timeout_error(prediction_id)

    def _prepare(
        self,
        model: str,
        prompt: str,
        resolved: _ResolvedParams,
        optional_params: Mapping[str, object],
        extra_headers: Mapping[str, str] | None,
        logging_obj: LiteLLMLoggingObj,
    ) -> _PreparedRequest:
        headers: Final = self.config.validate_environment(
            headers=MappingProxyType({**(extra_headers or MappingProxyType({}))}),
            model=model,
            messages=(),
            optional_params=optional_params,
            litellm_params=resolved.litellm_params,
            api_key=resolved.api_key,
            api_base=resolved.api_base,
        )
        submit_url: Final = self.config.get_complete_url(
            api_base=resolved.api_base,
            api_key=resolved.api_key,
            model=model,
            optional_params=optional_params,
            litellm_params=resolved.litellm_params,
        )
        body: Final = self.config.transform_image_generation_request(
            model=model,
            prompt=prompt,
            optional_params=optional_params,
            litellm_params=resolved.litellm_params,
            headers=headers,
        )

        logging_obj.pre_call(
            input=prompt,
            api_key="",
            additional_args=to_request_payload(
                MappingProxyType({"complete_input_dict": body, "api_base": submit_url, "headers": headers})
            ),
        )

        return _PreparedRequest(headers=headers, submit_url=submit_url, body=body)

    def _transform(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ImageResponse,
        logging_obj: LiteLLMLoggingObj,
        prepared: _PreparedRequest,
        optional_params: Mapping[str, object],
        resolved: _ResolvedParams,
    ) -> ImageResponse:
        return self.config.transform_image_generation_response(
            model=model,
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=logging_obj,
            request_data=prepared.body,
            optional_params=optional_params,
            litellm_params=resolved.litellm_params,
            encoding=None,
        )


def _resolve_params(litellm_params: GenericLiteLLMParams | Mapping[str, object]) -> _ResolvedParams:
    if isinstance(litellm_params, Mapping):
        api_key: Final = litellm_params.get("api_key")
        api_base: Final = litellm_params.get("api_base")
        return _ResolvedParams(
            api_key=api_key if isinstance(api_key, str) else None,
            api_base=api_base if isinstance(api_base, str) else None,
            litellm_params=MappingProxyType(dict(litellm_params)),
        )
    return _ResolvedParams(
        api_key=litellm_params.api_key,
        api_base=litellm_params.api_base,
        litellm_params=MappingProxyType(dict(litellm_params)),
    )


def _record_poll_failure(consecutive_failures: int, prediction_id: str, error: Exception) -> int:
    next_count: Final = consecutive_failures + 1
    if next_count >= MAX_CONSECUTIVE_POLL_FAILURES:
        raise WaveSpeedError(
            status_code=500,
            message=(
                f"WaveSpeed result polling for prediction {prediction_id} failed {next_count} times in a row: {error}"
            ),
        )
    verbose_logger.debug("WaveSpeed poll attempt failed (%s/%s): %s", next_count, MAX_CONSECUTIVE_POLL_FAILURES, error)
    return next_count


def _timeout_error(prediction_id: str) -> WaveSpeedError:
    return WaveSpeedError(
        status_code=408,
        message=f"WaveSpeed prediction {prediction_id} did not finish within {DEFAULT_MAX_POLLING_TIME} seconds",
    )


wavespeed_image_generation: Final = WaveSpeedImageGeneration()
