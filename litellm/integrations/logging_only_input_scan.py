import asyncio
import itertools
import time
from collections.abc import Coroutine, Iterator, Mapping, Sequence
from types import MappingProxyType
from typing import Final, TypeAlias

from pydantic import TypeAdapter, ValidationError

from litellm._logging import verbose_logger
from litellm.constants import LITELLM_LOGGING_NO_UPSTREAM_LLM_CALL, PROXY_REJECTED_BEFORE_ROUTING_KEY
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.litellm_core_utils.core_helpers import independent_snapshot
from litellm.llms import get_guardrail_translation_mapping
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import CallTypes

InputScans: TypeAlias = Mapping[CustomGuardrail, "asyncio.Task[object]"]

_GUARDRAIL_INFORMATION_KEY: Final = "standard_logging_guardrail_information"
_METADATA_BUCKETS: Final = ("metadata", "litellm_metadata")
_SCRATCH_REQUEST_KEYS: Final = frozenset(
    {"model", "messages", "input", "tools", "litellm_call_id", *_METADATA_BUCKETS, "proxy_server_request"}
)
_REQUEST: Final = TypeAdapter(dict[str, object])
_VERDICTS: Final = TypeAdapter(tuple[object, ...])
_RUNNING_SCANS: Final[set["asyncio.Task[object]"]] = set()  # mutable-ok: strong refs keep a scan alive until it ends


def _as_request(value: object) -> dict[str, object] | None:  # mutable-ok: guardrails record their verdict into it
    try:
        return _REQUEST.validate_python(value)
    except ValidationError:
        return None


def _without_recorded_verdicts(value: object) -> object:
    bucket: Final = _as_request(value)
    if bucket is None:
        return value
    return {key: item for key, item in bucket.items() if key != _GUARDRAIL_INFORMATION_KEY}


def _scratch_request(data: Mapping[str, object]) -> dict[str, object]:  # mutable-ok: the scan records its verdict here
    selected: Final = {key: item for key, item in data.items() if key in _SCRATCH_REQUEST_KEYS}
    without_verdicts: Final[dict[str, object]] = {  # mutable-ok: independent_snapshot takes a dict
        key: _without_recorded_verdicts(item) for key, item in selected.items()
    }
    snapshot: Final = _as_request(independent_snapshot(without_verdicts))
    return snapshot if snapshot is not None else {}


def _bucket_verdicts(bucket: object) -> tuple[object, ...]:
    entries: Final = (_as_request(bucket) or {}).get(_GUARDRAIL_INFORMATION_KEY)
    try:
        return _VERDICTS.validate_python(entries) if isinstance(entries, list) else ()
    except ValidationError:
        return ()


def _recorded_verdicts(scratch_request: Mapping[str, object]) -> tuple[object, ...]:
    return tuple(itertools.chain.from_iterable(_bucket_verdicts(scratch_request.get(key)) for key in _METADATA_BUCKETS))


async def _scan_input(
    guardrail: CustomGuardrail,
    call_type: CallTypes,
    scratch_request: dict[str, object],  # mutable-ok: the guardrail records its verdict into the scratch request
) -> tuple[object, ...]:
    try:
        await get_guardrail_translation_mapping(call_type)().process_input_messages(
            data=scratch_request, guardrail_to_apply=guardrail
        )
    except Exception as e:  # noqa: BLE001  # an observe-only scan never affects the request; its verdict is already recorded
        verbose_logger.warning("Guardrail %s: logging_only on_arrival scan raised: %s", guardrail.guardrail_name, e)
    return _recorded_verdicts(scratch_request)


def _translatable_call_type(call_type: str) -> CallTypes | None:
    try:
        parsed: Final = CallTypes(call_type)
        get_guardrail_translation_mapping(parsed)
    except ValueError:
        return None
    return parsed


def _on_arrival_guardrails(callbacks: Sequence[object], request: Mapping[str, object]) -> Iterator[CustomGuardrail]:
    for callback in callbacks:
        if (
            isinstance(callback, CustomGuardrail)
            and callback.logging_only_input_timing == "on_arrival"
            and callback.should_run_guardrail(data=request, event_type=GuardrailEventHooks.logging_only) is True
        ):
            yield callback


def logging_only_decision_data(
    guardrail: CustomGuardrail, model_call_details: Mapping[str, object]
) -> Mapping[str, object]:
    """What a log-time should_run_guardrail(logging_only) check sees. A guardrail that scans through the generic hook
    gets the call's metadata buckets at the top level, so key and team opt-outs apply as they do at pre_call (Logging
    keeps them under litellm_params). A guardrail with its own logging hook, such as one that masks logs, keeps the
    call as is, so an opted-out key never turns its log redaction off."""
    if not guardrail.supports_logging_only_scope():
        return model_call_details
    litellm_params: Final = _as_request(model_call_details.get("litellm_params")) or {}
    lifted: Final = {bucket: litellm_params[bucket] for bucket in _METADATA_BUCKETS if litellm_params.get(bucket)}
    return {**model_call_details, **lifted}


def never_reached_a_model(model_call_details: Mapping[str, object]) -> bool:
    """Whether the failed call was rejected before any model was called, by a proxy gate or the router."""
    litellm_params: Final = _as_request(model_call_details.get("litellm_params")) or {}
    return bool(
        model_call_details.get(LITELLM_LOGGING_NO_UPSTREAM_LLM_CALL)
        or litellm_params.get(PROXY_REJECTED_BEFORE_ROUTING_KEY)
    )


def _log_scan_failure(task: "asyncio.Task[object]") -> None:
    if task.cancelled() or task.exception() is None:
        return
    verbose_logger.warning("%s raised: %s", task.get_name(), task.exception())


def spawn_scan(coroutine: Coroutine[object, object, object], name: str) -> "asyncio.Task[object]":
    task: Final = asyncio.get_running_loop().create_task(coroutine, name=name)
    _RUNNING_SCANS.add(task)
    task.add_done_callback(_RUNNING_SCANS.discard)
    task.add_done_callback(_log_scan_failure)
    return task


def start_input_scans(callbacks: Sequence[object], data: object, call_type: str) -> InputScans:
    """Send the request to every on_arrival logging_only guardrail in the background, without waiting."""
    request: Final = _as_request(data)
    guardrails: Final = tuple(_on_arrival_guardrails(callbacks, request)) if request is not None else ()
    translatable: Final = _translatable_call_type(call_type) if guardrails else None
    if request is None or translatable is None:
        return MappingProxyType({})
    return MappingProxyType(
        {
            guardrail: spawn_scan(
                _scan_input(guardrail, translatable, _scratch_request(request)),
                f"Guardrail {guardrail.guardrail_name} logging_only on_arrival scan",
            )
            for guardrail in guardrails
        }
    )


def _not_ready_verdict(guardrail: CustomGuardrail, reason: str) -> tuple[object, ...]:
    scratch_request: Final[dict[str, object]] = {"metadata": {}}  # mutable-ok: the guardrail records its entry here
    recorded_at: Final = time.time()
    guardrail.add_standard_logging_guardrail_information_to_request_data(
        guardrail_json_response=reason,
        request_data=scratch_request,
        guardrail_status="not_run",
        start_time=recorded_at,
        end_time=recorded_at,
        duration=0.0,
        event_type=GuardrailEventHooks.logging_only,
    )
    return _recorded_verdicts(scratch_request)


def _timed_out(guardrail: CustomGuardrail) -> str:
    return f"no verdict within {guardrail.logging_only_verdict_timeout:g}s"


def _finished_verdicts(guardrail: CustomGuardrail, scan: "asyncio.Task[object]") -> tuple[object, ...]:
    if scan.cancelled():
        return _not_ready_verdict(guardrail, _timed_out(guardrail))
    if scan.exception() is not None:
        return ()
    verdicts: Final = scan.result()
    return _VERDICTS.validate_python(verdicts) if isinstance(verdicts, tuple) else ()


async def collect_input_scan(guardrail: CustomGuardrail, scan: "asyncio.Task[object]") -> tuple[object, ...]:
    """The scan's verdicts, waiting at most the guardrail's verdict timeout and cancelling the scan after it."""
    if not scan.done():
        await asyncio.wait((scan,), timeout=guardrail.logging_only_verdict_timeout)
    if not scan.done():
        scan.cancel()
        return _not_ready_verdict(guardrail, _timed_out(guardrail))
    return _finished_verdicts(guardrail, scan)


def peek_input_scan(guardrail: CustomGuardrail, scan: "asyncio.Task[object]") -> tuple[object, ...]:
    """The scan's verdicts if it already finished, without waiting or cancelling, so a failed attempt never delays
    its error or a retry, and the scan stays alive for the attempt that succeeds."""
    if not scan.done():
        return _not_ready_verdict(guardrail, "no verdict yet when the call failed")
    return _finished_verdicts(guardrail, scan)
