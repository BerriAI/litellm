"""
Handler for the TinyFish Agent provider: /run-async + poll /v1/runs/{id} for non-streaming
(runs take minutes; blocking /run is fragile behind proxies, and poll-timeout cancels the run
so it stops billing), native /run-sse for streaming.
"""

import asyncio
import json
import time
from collections.abc import AsyncIterator, Mapping
from types import MappingProxyType
from typing import Final, NamedTuple
from uuid import uuid4

import httpx
from pydantic import TypeAdapter, ValidationError

from litellm._logging import verbose_logger
from litellm.a2a_protocol.providers.base import A2A_PROVIDER_RESPONSE_COST_KEY
from litellm.a2a_protocol.providers.tinyfish.transformation import (
    EMPTY_MAPPING,
    TERMINAL_RUN_STATUSES,
    TinyfishAgentTransformation,
    TinyfishRun,
    as_str_object_dict,
)
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    get_async_httpx_client,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.custom_http import httpxSpecialProvider

_DEFAULT_API_BASE: Final = "https://agent.tinyfish.ai"
_POLL_INTERVAL_S: Final = 2.0
_DEFAULT_POLLING_TIMEOUT_S: Final = 600.0

_RunAdapter: Final = TypeAdapter(TinyfishRun)
_FloatAdapter: Final = TypeAdapter(float)


class TinyfishRequestContext(NamedTuple):
    api_key: str
    api_base: str
    cost_per_step: float | None
    polling_timeout_s: float
    allow_authenticated_runs: bool
    default_request_params: Mapping[str, object]


class TinyfishAgentHandler:
    @staticmethod
    def _http_client(timeout: float) -> AsyncHTTPHandler:
        return get_async_httpx_client(
            llm_provider=httpxSpecialProvider.A2AProvider,
            params={"timeout": timeout},  # mutable-ok: get_async_httpx_client takes a plain dict of client params
        )

    @staticmethod
    def _extract_litellm_params(
        litellm_params: Mapping[str, object],
        api_base: str | None,
    ) -> TinyfishRequestContext:
        configured_key: Final = litellm_params.get("api_key")
        api_key: Final = (
            configured_key if isinstance(configured_key, str) and configured_key else get_secret_str("TINYFISH_API_KEY")
        )
        if not api_key:
            raise ValueError(
                "TinyFish Agent: no API key. Set `api_key` in the agent's litellm_params "
                "or the TINYFISH_API_KEY environment variable."
            )

        configured_base: Final = litellm_params.get("api_base")
        resolved_base: Final = (
            api_base
            or (configured_base if isinstance(configured_base, str) and configured_base else None)
            or get_secret_str("TINYFISH_AGENT_API_BASE")
            or _DEFAULT_API_BASE
        )

        cost_per_step: Final = _validated_float(litellm_params.get("cost_per_step"), "cost_per_step")
        polling_timeout: Final = _validated_float(
            litellm_params.get("polling_timeout_seconds"), "polling_timeout_seconds"
        )

        return TinyfishRequestContext(
            api_key=api_key,
            api_base=resolved_base.rstrip("/"),
            cost_per_step=cost_per_step,
            polling_timeout_s=polling_timeout if polling_timeout is not None else _DEFAULT_POLLING_TIMEOUT_S,
            allow_authenticated_runs=bool(litellm_params.get("allow_authenticated_runs")),
            default_request_params=_validated_request_defaults(litellm_params.get("default_request_params")),
        )

    @staticmethod
    def _auth_headers(api_key: str, accept: str = "application/json") -> dict[str, str]:  # mutable-ok: httpx dict param
        return {  # mutable-ok: AsyncHTTPHandler.get/post take a plain dict for headers
            "X-API-Key": api_key,
            "Content-Type": "application/json",
            "Accept": accept,
        }

    @staticmethod
    def _parse_run(payload: object) -> TinyfishRun:
        try:
            return _RunAdapter.validate_python(payload)
        except ValidationError as e:
            raise RuntimeError(TinyfishAgentTransformation.wrap_error_message(f"unexpected response shape: {e}")) from e

    @staticmethod
    async def _submit_run(
        ctx: TinyfishRequestContext,
        body: Mapping[str, object],
        client: AsyncHTTPHandler,
    ) -> str:
        try:
            response: Final = await client.post(
                f"{ctx.api_base}/v1/automation/run-async",
                json={**body},  # mutable-ok: httpx json= serializes via json.dumps, which needs a plain dict
                headers=TinyfishAgentHandler._auth_headers(ctx.api_key),
            )
        except httpx.HTTPStatusError as e:
            raise RuntimeError(TinyfishAgentTransformation.wrap_error_message(_http_error_body(e))) from e

        payload: Final[object] = response.json()  # any-ok: httpx Response.json() -> Any
        run: Final = TinyfishAgentHandler._parse_run(payload)
        error: Final = run.get("error")
        run_id: Final = run.get("run_id")
        if error or not run_id:
            raise RuntimeError(TinyfishAgentTransformation.run_failure_message(run))
        verbose_logger.info("TinyFish Agent: submitted run_id=%s", run_id)
        return run_id

    @staticmethod
    async def _get_run(
        ctx: TinyfishRequestContext,
        run_id: str,
        client: AsyncHTTPHandler,
    ) -> TinyfishRun:
        # screenshots=none keeps the poll payload small (no per-step screenshot URLs needed).
        response: Final = await client.get(
            f"{ctx.api_base}/v1/runs/{run_id}?screenshots=none",
            headers=TinyfishAgentHandler._auth_headers(ctx.api_key),
        )
        if not (200 <= response.status_code < 300):
            raise RuntimeError(TinyfishAgentTransformation.wrap_error_message(response.text))
        payload: Final[object] = response.json()  # any-ok: httpx Response.json() -> Any
        return TinyfishAgentHandler._parse_run(payload)

    @staticmethod
    async def _cancel_run(
        ctx: TinyfishRequestContext,
        run_id: str,
        client: AsyncHTTPHandler,
    ) -> None:
        try:
            await client.post(
                f"{ctx.api_base}/v1/runs/{run_id}/cancel",
                headers=TinyfishAgentHandler._auth_headers(ctx.api_key),
            )
        except Exception as e:  # noqa: BLE001  # best-effort cancel; the timeout error below matters more
            verbose_logger.warning("TinyFish Agent: failed to cancel run %s after timeout: %r", run_id, e)

    @staticmethod
    async def _poll_until_terminal(
        ctx: TinyfishRequestContext,
        run_id: str,
        client: AsyncHTTPHandler,
    ) -> TinyfishRun:
        deadline: Final = time.monotonic() + ctx.polling_timeout_s
        while time.monotonic() < deadline:
            await asyncio.sleep(_POLL_INTERVAL_S)
            run = await TinyfishAgentHandler._get_run(ctx, run_id, client)
            status = run.get("status") or ""
            verbose_logger.debug("TinyFish Agent: poll run=%s status=%s", run_id, status)
            if status in TERMINAL_RUN_STATUSES:
                return run

        await TinyfishAgentHandler._cancel_run(ctx, run_id, client)
        raise RuntimeError(
            TinyfishAgentTransformation.wrap_error_message(
                f"run {run_id} did not finish within polling_timeout_seconds={ctx.polling_timeout_s:.0f}; "
                "the run was cancelled"
            )
        )

    @staticmethod
    def _run_cost(ctx: TinyfishRequestContext, run: TinyfishRun) -> float | None:
        num_of_steps: Final = run.get("num_of_steps")
        if ctx.cost_per_step is None or num_of_steps is None:
            return None
        return num_of_steps * ctx.cost_per_step

    @staticmethod
    async def handle_non_streaming(
        request_id: str,
        params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        api_base: str | None = None,
    ) -> Mapping[str, object]:
        ctx: Final = TinyfishAgentHandler._extract_litellm_params(litellm_params, api_base)
        body: Final = TinyfishAgentTransformation.build_run_body(
            params=params,
            default_request_params=ctx.default_request_params,
            allow_authenticated_runs=ctx.allow_authenticated_runs,
        )
        client: Final = TinyfishAgentHandler._http_client(timeout=30.0)

        run_id: Final = await TinyfishAgentHandler._submit_run(ctx, body, client)
        run: Final = await TinyfishAgentHandler._poll_until_terminal(ctx, run_id, client)

        if run.get("status") != "COMPLETED":
            raise RuntimeError(TinyfishAgentTransformation.run_failure_message(run))

        response: Final = TinyfishAgentTransformation.build_a2a_message_response(request_id, run)
        return TinyfishAgentTransformation.with_response_cost(response, TinyfishAgentHandler._run_cost(ctx, run))

    @staticmethod
    async def handle_streaming(
        request_id: str,
        params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        api_base: str | None = None,
    ) -> AsyncIterator[Mapping[str, object]]:
        ctx: Final = TinyfishAgentHandler._extract_litellm_params(litellm_params, api_base)
        body: Final = TinyfishAgentTransformation.build_run_body(
            params=params,
            default_request_params=ctx.default_request_params,
            allow_authenticated_runs=ctx.allow_authenticated_runs,
        )
        client: Final = TinyfishAgentHandler._http_client(timeout=ctx.polling_timeout_s)

        try:
            response: Final = await client.post(
                f"{ctx.api_base}/v1/automation/run-sse",
                json={**body},  # mutable-ok: httpx json= serializes via json.dumps, which needs a plain dict
                headers=TinyfishAgentHandler._auth_headers(ctx.api_key, accept="text/event-stream"),
                stream=True,
            )
        except httpx.HTTPStatusError as e:
            raise RuntimeError(TinyfishAgentTransformation.wrap_error_message(_http_error_body(e))) from e
        except httpx.TransportError as exc:
            verbose_logger.warning(
                "TinyFish Agent: SSE request failed before a run was submitted (%r); "
                "falling back to run-async + synthetic stream",
                exc,
            )
            async for chunk in TinyfishAgentHandler._fallback_synthetic_stream(
                request_id=request_id,
                params=params,
                litellm_params=litellm_params,
                api_base=api_base,
            ):
                yield chunk
            return

        context_id: Final = str(uuid4())
        task_id = ""  # rebind-ok: filled in by the first SSE event that carries a run_id
        saw_terminal = False  # rebind-ok: flipped when the COMPLETE event arrives
        try:
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if not data_str:
                    continue
                try:
                    event_payload: object = json.loads(data_str)  # any-ok: json.loads -> Any
                    event = TinyfishAgentHandler._parse_run(event_payload)
                except (json.JSONDecodeError, RuntimeError):
                    verbose_logger.debug("TinyFish Agent: skipping unparseable SSE line")
                    continue

                event_type = event.get("type") or ""
                task_id = event.get("run_id") or task_id or str(uuid4())

                if event_type == "STARTED":
                    yield TinyfishAgentTransformation.task_submitted_event(request_id, task_id, context_id)
                elif event_type == "STREAMING_URL":
                    streaming_url = event.get("streaming_url") or ""
                    yield TinyfishAgentTransformation.working_status_event(
                        request_id, task_id, context_id, f"Watch live: {streaming_url}"
                    )
                elif event_type == "PROGRESS":
                    yield TinyfishAgentTransformation.working_status_event(
                        request_id, task_id, context_id, event.get("purpose") or ""
                    )
                elif event_type == "COMPLETE":
                    saw_terminal = True
                    async for chunk in TinyfishAgentHandler._terminal_events(
                        ctx=ctx,
                        request_id=request_id,
                        task_id=task_id,
                        context_id=context_id,
                        event=event,
                        client=client,
                    ):
                        yield chunk
        finally:
            if hasattr(response, "aclose"):
                await response.aclose()

        if saw_terminal:
            return
        if not task_id:
            raise RuntimeError(
                TinyfishAgentTransformation.wrap_error_message("SSE stream ended before a run was started")
            )
        # TinyFish SSE has no reconnection; the run may still finish server-side.
        yield TinyfishAgentTransformation.final_status_event(
            request_id,
            task_id,
            context_id,
            state="failed",
            message_text=TinyfishAgentTransformation.wrap_error_message(
                f"SSE stream ended before the run finished; check GET /v1/runs/{task_id} for the final result"
            ),
        )

    @staticmethod
    async def _run_with_steps(
        ctx: TinyfishRequestContext,
        task_id: str,
        client: AsyncHTTPHandler,
        event: TinyfishRun,
    ) -> TinyfishRun:
        """The COMPLETE event carries no num_of_steps; one run fetch prices the run."""
        if ctx.cost_per_step is None or not task_id:
            return event
        try:
            return await TinyfishAgentHandler._get_run(ctx, task_id, client)
        except Exception as e:  # noqa: BLE001  # never fail a finished stream over cost lookup
            verbose_logger.warning("TinyFish Agent: could not fetch run %s for cost: %r", task_id, e)
            return event

    @staticmethod
    async def _terminal_events(
        ctx: TinyfishRequestContext,
        request_id: str,
        task_id: str,
        context_id: str,
        event: TinyfishRun,
        client: AsyncHTTPHandler,
    ) -> AsyncIterator[Mapping[str, object]]:
        """Emit artifact + final status for a COMPLETE SSE event, with actual-cost lookup."""
        run: Final = await TinyfishAgentHandler._run_with_steps(ctx, task_id, client, event)
        cost: Final = TinyfishAgentHandler._run_cost(ctx, run)

        if event.get("status") == "COMPLETED":
            # The COMPLETE event's result is authoritative; the fetched run adds num_of_steps.
            artifact_run: Final = _RunAdapter.validate_python(
                MappingProxyType({**run, "result": event.get("result") if "result" in event else run.get("result")})
            )
            yield TinyfishAgentTransformation.artifact_event(request_id, task_id, context_id, artifact_run)
            yield TinyfishAgentTransformation.with_response_cost(
                TinyfishAgentTransformation.final_status_event(request_id, task_id, context_id, state="completed"),
                cost,
            )
            return

        yield TinyfishAgentTransformation.with_response_cost(
            TinyfishAgentTransformation.final_status_event(
                request_id,
                task_id,
                context_id,
                state="failed",
                message_text=TinyfishAgentTransformation.run_failure_message(event),
            ),
            cost,
        )

    @staticmethod
    async def _fallback_synthetic_stream(
        request_id: str,
        params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        api_base: str | None,
    ) -> AsyncIterator[Mapping[str, object]]:
        response: Final = await TinyfishAgentHandler.handle_non_streaming(
            request_id=request_id,
            params=params,
            litellm_params=litellm_params,
            api_base=api_base,
        )
        raw_cost: Final = response.get(A2A_PROVIDER_RESPONSE_COST_KEY)
        cost: Final = float(raw_cost) if isinstance(raw_cost, (int, float)) else None
        result: Final = as_str_object_dict(response.get("result")) or EMPTY_MAPPING

        task_id: Final = str(uuid4())
        context_id: Final = str(uuid4())
        yield TinyfishAgentTransformation.task_submitted_event(request_id, task_id, context_id)
        yield TinyfishAgentTransformation.artifact_event_from_parts(
            request_id=request_id,
            task_id=task_id,
            context_id=context_id,
            parts=result.get("parts", ()),
            metadata=result.get("metadata", EMPTY_MAPPING),
        )
        yield TinyfishAgentTransformation.with_response_cost(
            TinyfishAgentTransformation.final_status_event(request_id, task_id, context_id, state="completed"),
            cost,
        )


def _validated_float(value: object, field_name: str) -> float | None:
    if value is None:
        return None
    try:
        return _FloatAdapter.validate_python(value)
    except ValidationError:
        raise ValueError(f"TinyFish Agent: `{field_name}` in litellm_params must be a number, got {value!r}.")


def _validated_request_defaults(value: object) -> Mapping[str, object]:
    if value is None:
        return EMPTY_MAPPING
    validated: Final = as_str_object_dict(value)
    if validated is None:
        raise ValueError("TinyFish Agent: `default_request_params` in litellm_params must be an object.")
    return validated


def _http_error_body(e: httpx.HTTPStatusError) -> str:
    masked_body: Final = getattr(e, "message", None)
    if isinstance(masked_body, str) and masked_body:
        return masked_body
    return str(e)
