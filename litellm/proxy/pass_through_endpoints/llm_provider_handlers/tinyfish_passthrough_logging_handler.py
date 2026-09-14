import asyncio
import json
import os
import time
import urllib.parse
from collections.abc import Mapping, Sequence
from datetime import datetime
from types import MappingProxyType
from typing import Final, NamedTuple
from urllib.parse import urlparse

import httpx
from pydantic import TypeAdapter, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.litellm_logging import (
    get_standard_logging_object_payload,
)
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    get_async_httpx_client,
)
from litellm.proxy._types import PassThroughEndpointLoggingTypedDict
from litellm.types.llms.custom_http import httpxSpecialProvider
from litellm.types.passthrough_endpoints.tinyfish import (
    TINYFISH_AGENT_DEFAULT_API_BASE,
    TINYFISH_DEFAULT_COST_PER_STEP,
    TINYFISH_MAX_POLLING_SECONDS,
    TINYFISH_MODEL_NAME,
    TINYFISH_POLLING_INTERVAL_SECONDS,
    TINYFISH_TERMINAL_RUN_STATUSES,
    TinyfishRun,
)
from litellm.types.utils import StandardPassThroughResponseObject

_RUN_ADAPTER: Final = TypeAdapter(TinyfishRun)

_EMPTY_KWARGS: Final[Mapping[str, object]] = MappingProxyType({})


class _TinyfishLoggingPayload(NamedTuple):
    result: StandardPassThroughResponseObject
    kwargs: Mapping[str, object]

    def as_handler_result(self) -> PassThroughEndpointLoggingTypedDict:
        handler_result: Final[PassThroughEndpointLoggingTypedDict] = {
            "result": self.result,
            "kwargs": {**self.kwargs},
        }
        return handler_result


# asyncio tasks are weakly referenced by the loop; hold them until done or they can vanish mid-poll
_BACKGROUND_BILLING_TASKS: Final[set["asyncio.Task[None]"]] = set()  # mutable-ok: task registry


def resolve_tinyfish_agent_api_base() -> str:
    return (os.getenv("TINYFISH_AGENT_API_BASE") or TINYFISH_AGENT_DEFAULT_API_BASE).rstrip("/")


def resolve_tinyfish_cost_per_step() -> float:
    raw: Final = os.getenv("TINYFISH_COST_PER_STEP")
    if raw is None:
        return TINYFISH_DEFAULT_COST_PER_STEP
    try:
        return float(raw)
    except ValueError:
        verbose_proxy_logger.warning(
            "TINYFISH_COST_PER_STEP=%r is not a number; using the default rate %s",
            raw,
            TINYFISH_DEFAULT_COST_PER_STEP,
        )
        return TINYFISH_DEFAULT_COST_PER_STEP


def is_tinyfish_agent_url(url: str) -> bool:
    hostname: Final = urlparse(url).hostname
    return hostname is not None and hostname == urlparse(resolve_tinyfish_agent_api_base()).hostname


def _parse_run(payload: object) -> TinyfishRun | None:
    try:
        return _RUN_ADAPTER.validate_python(payload)
    except ValidationError as e:
        verbose_proxy_logger.warning("TinyFish passthrough: unexpected run object shape: %s", e)
        return None


def _run_cost(run: TinyfishRun | None) -> float | None:
    if run is None:
        return None
    num_of_steps: Final = run.get("num_of_steps")
    if num_of_steps is None:
        return None
    return num_of_steps * resolve_tinyfish_cost_per_step()


class TinyFishPassthroughLoggingHandler:
    @staticmethod
    def _should_log_request(request_method: str, url_route: str) -> bool:
        """Only run submissions are billed; GET /v1/runs* polling and cancels never write spend rows."""
        return request_method == "POST" and "/v1/automation/" in urlparse(url_route).path

    @staticmethod
    def is_run_async_route(url_route: str) -> bool:
        return urlparse(url_route).path.endswith("/v1/automation/run-async")

    @staticmethod
    def tinyfish_passthrough_handler(
        httpx_response: httpx.Response,
        response_body: Mapping[str, object] | None,
        logging_obj: LiteLLMLoggingObj,
        url_route: str,
        result: str,
        start_time: datetime,
        end_time: datetime,
        cache_hit: bool,
        request_body: Mapping[str, object],
        **kwargs: object,  # kwargs-ok: the passthrough logging dispatch forwards shared logging kwargs to every handler
    ) -> PassThroughEndpointLoggingTypedDict:
        """Bill a blocking POST /v1/automation/run: the response is the terminal run object."""
        try:
            run: Final = _parse_run(response_body) if response_body is not None else None
            handler_payload: Final = TinyFishPassthroughLoggingHandler._build_logging_payload(
                run=run,
                logging_obj=logging_obj,
                result=result,
                start_time=start_time,
                end_time=end_time,
                kwargs=kwargs,
            ).as_handler_result()
        except Exception as e:
            verbose_proxy_logger.exception("Error in TinyFish passthrough logging handler: %s", e)
            fallback_payload: Final[PassThroughEndpointLoggingTypedDict] = {
                "result": StandardPassThroughResponseObject(response=result),
                "kwargs": kwargs,
            }
            return fallback_payload
        return handler_payload

    @staticmethod
    def start_async_run_billing(
        response_body: Mapping[str, object] | None,
        logging_obj: LiteLLMLoggingObj,
        result: str,
        start_time: datetime,
        cache_hit: bool,
        **kwargs: object,  # kwargs-ok: shared logging kwargs, replayed into _handle_logging when the run finishes
    ) -> None:
        """Bill POST /v1/automation/run-async once, when the polled run turns terminal."""
        submitted: Final = _parse_run(response_body) if response_body is not None else None
        run_id: Final = submitted.get("run_id") if submitted is not None else None
        if not run_id:
            verbose_proxy_logger.warning(
                "TinyFish passthrough: run-async response carried no run_id; logging the request without cost"
            )
        task: Final = asyncio.create_task(
            TinyFishPassthroughLoggingHandler._poll_and_log(
                run_id=run_id,
                logging_obj=logging_obj,
                result=result,
                start_time=start_time,
                cache_hit=cache_hit,
                kwargs=kwargs,
            )
        )
        _BACKGROUND_BILLING_TASKS.add(task)
        task.add_done_callback(_BACKGROUND_BILLING_TASKS.discard)

    @staticmethod
    async def _poll_and_log(
        run_id: str | None,
        logging_obj: LiteLLMLoggingObj,
        result: str,
        start_time: datetime,
        cache_hit: bool,
        kwargs: Mapping[str, object],
        client: AsyncHTTPHandler | None = None,
    ) -> None:
        from ..pass_through_endpoints import pass_through_endpoint_logging

        try:
            run: Final = (
                await TinyFishPassthroughLoggingHandler._poll_until_terminal(run_id, client) if run_id else None
            )
            run_end_time: Final = datetime.now()  # noqa: DTZ005  # naive to match the start_time stamped by pass_through_request
            payload: Final = TinyFishPassthroughLoggingHandler._build_logging_payload(
                run=run,
                logging_obj=logging_obj,
                result=result,
                start_time=start_time,
                end_time=run_end_time,
                kwargs=kwargs,
            )
            await pass_through_endpoint_logging._handle_logging(  # pyright: ignore[reportPrivateUsage]  # shared passthrough logging dispatcher, same access as the assemblyai handler
                logging_obj=logging_obj,
                standard_logging_response_object=payload.result,
                result=result,
                start_time=start_time,
                end_time=run_end_time,
                cache_hit=cache_hit,
                **payload.kwargs,
            )
        except Exception as e:
            verbose_proxy_logger.exception("[Non blocking logging error] TinyFish run-async billing failed: %s", e)

    @staticmethod
    async def _poll_until_terminal(run_id: str, client: AsyncHTTPHandler | None = None) -> TinyfishRun | None:
        deadline: Final = time.monotonic() + TINYFISH_MAX_POLLING_SECONDS
        while time.monotonic() < deadline:
            run = await TinyFishPassthroughLoggingHandler._fetch_run(run_id, client)
            if run is None:
                return None
            if (run.get("status") or "") in TINYFISH_TERMINAL_RUN_STATUSES:
                return run
            await asyncio.sleep(TINYFISH_POLLING_INTERVAL_SECONDS)
        verbose_proxy_logger.warning(
            "TinyFish passthrough: run %s not terminal after %ss; logging the request without cost",
            run_id,
            TINYFISH_MAX_POLLING_SECONDS,
        )
        return None

    @staticmethod
    async def _fetch_run(run_id: str, client: AsyncHTTPHandler | None = None) -> TinyfishRun | None:
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            passthrough_endpoint_router,
        )

        api_key: Final = passthrough_endpoint_router.get_credentials(custom_llm_provider="tinyfish", region_name=None)
        if api_key is None:
            verbose_proxy_logger.warning("TinyFish passthrough: no API key available to poll run %s", run_id)
            return None
        if any(c in run_id for c in ("/", "\\", "#", "?")) or ".." in run_id:
            verbose_proxy_logger.warning("TinyFish passthrough: invalid run_id %r", run_id)
            return None
        safe_run_id: Final = urllib.parse.quote(run_id, safe="")
        resolved_client: Final = client or get_async_httpx_client(
            llm_provider=httpxSpecialProvider.PassThroughEndpoint,
            params={"timeout": 30.0},  # mutable-ok: get_async_httpx_client takes a plain dict of client params
        )
        try:
            # screenshots=none keeps the poll payload small (no per-step screenshot URLs needed)
            response: Final = await resolved_client.get(
                f"{resolve_tinyfish_agent_api_base()}/v1/runs/{safe_run_id}?screenshots=none",
                headers={"X-API-Key": api_key},  # mutable-ok: httpx headers= takes a plain dict
            )
            if not (200 <= response.status_code < 300):
                verbose_proxy_logger.warning(
                    "TinyFish passthrough: GET /v1/runs/%s returned %s", safe_run_id, response.status_code
                )
                return None
            payload: Final[object] = response.json()  # any-ok: httpx Response.json() -> Any
            return _parse_run(payload)
        except Exception as e:
            verbose_proxy_logger.warning("[Non blocking logging error] TinyFish run fetch failed: %s", e)
            return None

    @staticmethod
    async def _handle_logging_tinyfish_collected_chunks(
        litellm_logging_obj: LiteLLMLoggingObj,
        url_route: str,
        start_time: datetime,
        all_chunks: Sequence[str],
        end_time: datetime,
        client: AsyncHTTPHandler | None = None,
    ) -> PassThroughEndpointLoggingTypedDict:
        """Bill a POST /v1/automation/run-sse stream: SSE events carry no num_of_steps, so the
        run_id parsed from the buffered events prices the run via one GET /v1/runs/{id}."""
        try:
            run_id: Final = _run_id_from_sse_chunks(all_chunks)
            if run_id is None:
                verbose_proxy_logger.warning(
                    "TinyFish passthrough: no run_id in SSE stream; logging the request without cost"
                )
            run: Final = await TinyFishPassthroughLoggingHandler._fetch_run(run_id, client) if run_id else None
            payload: Final = TinyFishPassthroughLoggingHandler._build_logging_payload(
                run=run,
                logging_obj=litellm_logging_obj,
                result="",
                start_time=start_time,
                end_time=end_time,
                kwargs=_EMPTY_KWARGS,
            ).as_handler_result()
        except Exception as e:
            verbose_proxy_logger.exception("Error in TinyFish SSE passthrough logging handler: %s", e)
            fallback_payload: Final[PassThroughEndpointLoggingTypedDict] = {
                "result": StandardPassThroughResponseObject(response=""),
                "kwargs": {},
            }
            return fallback_payload
        return payload

    @staticmethod
    def _build_logging_payload(
        run: TinyfishRun | None,
        logging_obj: LiteLLMLoggingObj,
        result: str,
        start_time: datetime,
        end_time: datetime,
        kwargs: Mapping[str, object],
    ) -> _TinyfishLoggingPayload:
        response_cost: Final = _run_cost(run)
        updated_kwargs: Final = {  # mutable-ok: the logging pipeline requires a plain kwargs dict
            **kwargs,
            "model": TINYFISH_MODEL_NAME,
            "custom_llm_provider": "tinyfish",
            "response_cost": response_cost,
        }
        logging_obj.model_call_details.update(
            model=TINYFISH_MODEL_NAME,
            custom_llm_provider="tinyfish",
            response_cost=response_cost,
        )

        logged_response: Final = StandardPassThroughResponseObject(
            response=json.dumps(run) if run is not None else result
        )
        standard_logging_object: Final = get_standard_logging_object_payload(
            kwargs=updated_kwargs,
            init_response_obj=logged_response,
            start_time=start_time,
            end_time=end_time,
            logging_obj=logging_obj,
            status="success",
        )
        return _TinyfishLoggingPayload(
            result=logged_response,
            kwargs=MappingProxyType({**updated_kwargs, "standard_logging_object": standard_logging_object}),
        )


def _run_id_from_sse_chunks(all_chunks: Sequence[str]) -> str | None:
    for line in all_chunks:
        if not line.startswith("data:"):
            continue
        try:
            event_payload: object = json.loads(line[5:].strip())  # any-ok: json.loads -> Any
        except json.JSONDecodeError:
            continue
        event = _parse_run(event_payload)
        if event is None:
            continue
        run_id = event.get("run_id")
        if run_id:
            return run_id
    return None
