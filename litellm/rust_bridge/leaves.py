"""Leaf helpers invoked by the native callback dispatcher.

Rust selects every target, delivery and sequence. Nothing here chooses a callback,
reads a registry, or decides whether an event fires. Each function performs one
integration-specific or interop-specific action on values Rust hands it.

Labeled leaf helpers pending native migration (see rust-callback-inventory.md):
`prepare_success_logging`, `prepare_failure_logging`, `dispatch_named_success`,
`dispatch_named_failure`, `dispatch_callable`. They are deleted when cost and
payload construction move into core and when each string integration becomes a
CustomLogger.
"""

from __future__ import annotations

import datetime
import json
import traceback
from collections.abc import Awaitable, Callable, Mapping
from typing import (  # noqa: TID251  # narrows the untyped legacy Logging and CustomLogger surfaces once
    TYPE_CHECKING,
    Final,
    Literal,
    Protocol,
    cast,
)

from litellm.integrations.custom_logger import CustomLogger

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging

TerminalFamily = Literal["sync_success", "async_success", "sync_failure", "async_failure"]
Family = Literal["request", TerminalFamily]
Details = dict[str, object]
Timestamp = datetime.datetime
LegacyCall = Callable[..., object]
LegacyAsyncCall = Callable[..., Awaitable[None]]


class LoggerView(Protocol):
    model: str | None
    messages: object
    call_type: str
    start_time: Timestamp
    litellm_call_id: str
    completion_start_time: Timestamp | None
    model_call_details: Details
    log_raw_request_response: bool
    standard_callback_dynamic_params: object
    standard_built_in_tools_params: object

    def record_api_call_start_time(self) -> None: ...

    def record_post_call(
        self, original_response: object, input: object, api_key: object, additional_args: Details
    ) -> None: ...

    def should_run_callback(self, callback: object, litellm_params: Details, event_hook: str) -> bool: ...

    def _pre_call(self, input: str, api_key: str | None, model: str | None, additional_args: Details) -> None: ...

    def _print_llm_call_debugging_log(self, api_base: str, headers: Details, additional_args: Details) -> None: ...

    def _get_request_curl_command(
        self, api_base: str, headers: Details | None, additional_args: Details, data: object
    ) -> str: ...

    def _get_masked_api_base(self, api_base: str) -> str: ...

    def _get_raw_request_body(self, data: object) -> Details: ...

    def _get_masked_headers(self, headers: Details) -> Details: ...

    def _response_cost_calculator(self, result: object) -> float | None: ...

    def _build_standard_logging_payload(
        self, init_response_obj: object, start_time: Timestamp, end_time: Timestamp
    ) -> object: ...

    def _handle_callback_failure(self, callback: object) -> None: ...


class IntegrationView(Protocol):
    def log_pre_api_call(self, model: str | None, messages: object, kwargs: Details) -> None: ...

    def log_post_api_call(
        self, kwargs: Details, response_obj: object, start_time: Timestamp, end_time: Timestamp | None
    ) -> None: ...

    def log_success_event(
        self, kwargs: Details, response_obj: object, start_time: Timestamp, end_time: Timestamp
    ) -> None: ...

    def log_failure_event(
        self, kwargs: Details, response_obj: object, start_time: Timestamp, end_time: Timestamp
    ) -> None: ...

    def async_log_success_event(
        self, kwargs: Details, response_obj: object, start_time: Timestamp, end_time: Timestamp
    ) -> Awaitable[None]: ...

    def async_log_failure_event(
        self, kwargs: Details, response_obj: object, start_time: Timestamp, end_time: Timestamp
    ) -> Awaitable[None]: ...

    def logging_hook(self, kwargs: Details, result: object, call_type: str) -> tuple[Details, object]: ...

    def async_logging_hook(
        self, kwargs: Details, result: object, call_type: str
    ) -> Awaitable[tuple[Details, object]]: ...

    def redact_standard_logging_payload_from_model_call_details(self, model_call_details: Details) -> Details: ...

    def log_input_event(
        self, model: str | None, messages: object, kwargs: Details, print_verbose: LegacyCall, callback_func: LegacyCall
    ) -> None: ...

    def log_event(
        self,
        kwargs: Details,
        response_obj: object,
        start_time: Timestamp,
        end_time: Timestamp,
        print_verbose: LegacyCall,
        callback_func: LegacyCall,
    ) -> None: ...

    def async_log_event(
        self,
        kwargs: Details,
        response_obj: object,
        start_time: Timestamp,
        end_time: Timestamp,
        print_verbose: LegacyCall,
        callback_func: LegacyCall,
    ) -> Awaitable[None]: ...


def _logger(logger: Logging) -> LoggerView:
    return cast(LoggerView, logger)  # cast-ok: legacy Logging is untyped; this protocol names the attributes we read


def _integration(callback: CustomLogger) -> IntegrationView:
    return cast(IntegrationView, callback)  # cast-ok: legacy CustomLogger methods are untyped


def _legacy_module() -> Mapping[str, object]:
    from litellm.litellm_core_utils import litellm_logging

    return cast(
        Mapping[str, object], vars(litellm_logging)
    )  # cast-ok: module globals hold the legacy integration singletons


def _print_verbose() -> LegacyCall:
    from litellm.litellm_core_utils import litellm_logging

    return cast(LegacyCall, litellm_logging.print_verbose)  # cast-ok: legacy debug printer is untyped


def _method(target: object, name: str) -> LegacyCall:
    return cast(LegacyCall, getattr(target, name))  # cast-ok: legacy integration singletons are untyped


def _async_method(target: object, name: str) -> LegacyAsyncCall:
    return cast(LegacyAsyncCall, getattr(target, name))  # cast-ok: legacy integration singletons are untyped


def _redact_string(value: str) -> str:
    from litellm.litellm_core_utils import litellm_logging

    return cast(Callable[[str], str], litellm_logging._redact_string)(value)  # pyright: ignore[reportPrivateUsage]  # cast-ok: legacy helper


def _redact_result(details: Details, result: object) -> object:
    from litellm.litellm_core_utils import redact_messages

    redact: Final = cast(LegacyCall, redact_messages.redact_message_input_output_from_logging)  # cast-ok: legacy helper
    return redact(model_call_details=details, result=result)


def record_pre_call(
    logger: Logging,
    *,
    api_key: str | None,
    body: Details,
    headers: dict[str, str],
    url: str,
) -> None:
    view: Final = _logger(logger)
    additional_args: Final[Details] = {"complete_input_dict": body, "headers": headers, "api_base": url}
    view._pre_call(input="OCR document processing", api_key=api_key, model=None, additional_args=additional_args)  # pyright: ignore[reportPrivateUsage]  # legacy state writer
    view._print_llm_call_debugging_log(api_base=url, headers=dict(headers), additional_args=additional_args)  # pyright: ignore[reportPrivateUsage]  # legacy debug output
    _capture_raw_request(view, additional_args)
    _run_logger_fn(logger)
    view.record_api_call_start_time()


def _capture_raw_request(view: LoggerView, additional_args: Details) -> None:
    import litellm
    from litellm.types.utils import RawRequestTypedDict

    if not (view.log_raw_request_response or litellm.log_raw_request_response):
        return
    details: Final = view.model_call_details
    params: Final = cast(Details, details.get("litellm_params") or {})  # cast-ok: legacy nested dict
    metadata: Final = cast(Details, params.get("metadata") or {})  # cast-ok: legacy nested dict
    params.setdefault("metadata", metadata)
    if litellm.turn_off_message_logging:
        metadata["raw_request"] = "redacted by litellm. 'litellm.turn_off_message_logging=True'"
        return
    api_base: Final = str(additional_args.get("api_base") or "")
    headers: Final = cast(Details, additional_args.get("headers") or {})  # cast-ok: legacy nested dict
    body: Final = additional_args.get("complete_input_dict", {})
    try:
        curl: Final = view._get_request_curl_command(  # pyright: ignore[reportPrivateUsage]  # legacy debug formatter
            api_base=api_base, headers=headers, additional_args=additional_args, data=body
        )
        metadata["raw_request"] = _redact_string(str(curl))
        details["raw_request_typed_dict"] = RawRequestTypedDict(
            raw_request_api_base=view._get_masked_api_base(api_base),  # pyright: ignore[reportPrivateUsage]  # legacy masking
            raw_request_body=view._get_raw_request_body(body),  # pyright: ignore[reportPrivateUsage]  # legacy masking
            raw_request_headers=view._get_masked_headers(headers),  # pyright: ignore[reportPrivateUsage]  # legacy masking
            error=None,
        )
    except Exception as error:  # noqa: BLE001  # raw-request capture is best effort by contract
        details["raw_request_typed_dict"] = RawRequestTypedDict(error=str(error))
        metadata["raw_request"] = _redact_string(f"Unable to Log raw request: {error}")


def _run_logger_fn(logger: Logging) -> None:
    from litellm._logging import verbose_logger

    logger_fn: Final = cast(
        Callable[[Details], object] | None, getattr(logger, "logger_fn", None)
    )  # cast-ok: user hook is untyped
    if not callable(logger_fn):
        return
    try:
        logger_fn(_logger(logger).model_call_details)
    except Exception as error:  # noqa: BLE001  # user logger_fn failures never fail the request
        verbose_logger.exception("LiteLLM.LoggingError: [Non-Blocking] Exception occurred while logging %s", error)


def record_post_call(logger: Logging, *, original_response: object, body: Details, headers: dict[str, str]) -> None:
    view: Final = _logger(logger)
    serialized: Final = (
        json.dumps(original_response, default=str) if isinstance(original_response, dict) else original_response
    )
    view.record_post_call(
        original_response=serialized,
        input=None,
        api_key=None,
        additional_args={"complete_input_dict": body, "headers": headers},
    )
    _run_logger_fn(logger)
    _redact_result(view.model_call_details, serialized)


def log_pre_api_call(logger: Logging, callback: CustomLogger) -> None:
    view: Final = _logger(logger)
    _integration(callback).log_pre_api_call(model=view.model, messages=view.messages, kwargs=view.model_call_details)


def log_post_api_call(logger: Logging, callback: CustomLogger) -> None:
    view: Final = _logger(logger)
    _integration(callback).log_post_api_call(
        kwargs=view.model_call_details, response_obj=None, start_time=view.start_time, end_time=None
    )


def dispatch_named_request(logger: Logging, name: str, event: Literal["pre_api_call", "post_api_call"]) -> None:
    view: Final = _logger(logger)
    module: Final = _legacy_module()
    if name == "supabase" and event == "pre_api_call" and (client := module.get("supabaseClient")) is not None:
        details: Final = view.model_call_details
        _method(client, "input_log_event")(
            model=view.model,
            messages=view.messages,
            end_user=details.get("user", "default"),
            litellm_call_id=details["litellm_call_id"],
            print_verbose=_print_verbose(),
        )
    if name == "sentry" and (add_breadcrumb := module.get("add_breadcrumb")) is not None:
        cast(LegacyCall, add_breadcrumb)(  # cast-ok: legacy sentry hook
            category="litellm.llm_call", message=f"Model Call Details {event}: {view.model_call_details}", level="info"
        )


def dispatch_callable_request(logger: Logging, callback: LegacyCall) -> None:
    custom: Final = _legacy_module().get("customLogger")
    if not isinstance(custom, CustomLogger):
        return
    view: Final = _logger(logger)
    _integration(custom).log_input_event(
        model=view.model,
        messages=view.messages,
        kwargs=view.model_call_details,
        print_verbose=_print_verbose(),
        callback_func=callback,
    )


def report_target_failure(logger: Logging, callback: object, family: Family, error: BaseException) -> None:
    from litellm._logging import verbose_logger

    verbose_logger.error(
        "LiteLLM.LoggingError: [Non-Blocking] Exception occurred while %s logging with %s: %s",
        family,
        callback,
        "".join(traceback.format_exception(error)),
    )
    capture: Final = _legacy_module().get("capture_exception")
    if capture is not None and family in ("request", "sync_success", "sync_failure"):
        cast(LegacyCall, capture)(error)  # cast-ok: legacy sentry hook
    if family not in ("request", "sync_failure"):
        _logger(logger)._handle_callback_failure(callback=callback)  # pyright: ignore[reportPrivateUsage]  # legacy prometheus counter


def prepare_success_logging(logger: Logging, response: object, start_time: Timestamp, end_time: Timestamp) -> object:
    from litellm.litellm_core_utils.litellm_logging import emit_standard_logging_payload
    from litellm.types.utils import StandardLoggingPayload

    view: Final = _logger(logger)
    details: Final = view.model_call_details
    if view.completion_start_time is None:
        view.completion_start_time = end_time
        details["completion_start_time"] = end_time
    details["log_event_type"] = "successful_api_call"
    details["end_time"] = end_time
    details["cache_hit"] = None
    hidden: Final = cast(Details, getattr(response, "_hidden_params", None) or {})  # cast-ok: legacy response attribute
    params: Final = cast(Details | None, details.get("litellm_params"))  # cast-ok: legacy nested dict
    if hidden and params is not None:
        metadata: Final = cast(Details, params.get("metadata") or {})  # cast-ok: legacy nested dict
        params["metadata"] = metadata
        metadata["hidden_params"] = hidden
    existing: Final = details.get("response_cost")
    if "response_cost" in hidden:
        details["response_cost"] = hidden["response_cost"]
    elif existing is None or existing == 0:
        details["response_cost"] = view._response_cost_calculator(result=response)  # pyright: ignore[reportPrivateUsage]  # labeled leaf: native cost pending
    payload: Final = view._build_standard_logging_payload(response, start_time, end_time)  # pyright: ignore[reportPrivateUsage]  # labeled leaf: native payload pending
    details["standard_logging_object"] = payload
    if payload is not None:
        emit_standard_logging_payload(
            cast(StandardLoggingPayload, payload)
        )  # cast-ok: legacy builder returns the payload TypedDict
    return _redact_result(details, response)


def prepare_failure_logging(
    logger: Logging, exception: BaseException, start_time: Timestamp, end_time: Timestamp
) -> str:
    from litellm.litellm_core_utils import litellm_logging

    formatted: Final = "".join(traceback.format_exception(exception))
    view: Final = _logger(logger)
    details: Final = view.model_call_details
    if details.get("exception") is exception and details.get("standard_logging_object") is not None:
        return formatted
    details["log_event_type"] = "failed_api_call"
    details["exception"] = exception
    details["traceback_exception"] = _redact_string(formatted)
    details["end_time"] = end_time
    details.setdefault("original_response", None)
    if details.get("combined_usage_object") is None:
        details["response_cost"] = 0
    headers: Final = getattr(exception, "headers", None)
    if isinstance(headers, dict):
        params: Final = cast(Details, details.setdefault("litellm_params", {}))  # cast-ok: legacy nested dict
        metadata: Final = cast(Details, params.get("metadata") or {})  # cast-ok: legacy nested dict
        metadata.update(cast(Details, headers))  # cast-ok: exception headers are a plain dict
    build: Final = cast(
        LegacyCall, litellm_logging.get_standard_logging_object_payload
    )  # cast-ok: labeled leaf: native payload pending
    details["standard_logging_object"] = build(
        kwargs=details,
        init_response_obj={},
        start_time=start_time,
        end_time=end_time,
        logging_obj=logger,
        status="failure",
        error_str=_redact_string(str(exception)),
        original_exception=exception,
        standard_built_in_tools_params=view.standard_built_in_tools_params,
    )
    return formatted


_EVENT_HOOKS: Final[Mapping[TerminalFamily, str]] = {
    "sync_success": "success_handler",
    "async_success": "async_success_handler",
    "sync_failure": "failure_handler",
    "async_failure": "async_failure_handler",
}


def should_run_callback(logger: Logging, callback: object, family: TerminalFamily) -> bool:
    view: Final = _logger(logger)
    params: Final = cast(Details, view.model_call_details.get("litellm_params") or {})  # cast-ok: legacy nested dict
    return view.should_run_callback(callback=callback, litellm_params=params, event_hook=_EVENT_HOOKS[family])


def should_run_guardrail_hook(logger: Logging, callback: object) -> bool:
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.types.guardrails import GuardrailEventHooks

    if not isinstance(callback, CustomGuardrail):
        return True
    decide: Final = cast(LegacyCall, callback.should_run_guardrail)  # cast-ok: legacy guardrail method is untyped
    return decide(data=_logger(logger).model_call_details, event_type=GuardrailEventHooks.logging_only) is True


def logging_hook(logger: Logging, callback: CustomLogger, result: object) -> object:
    view: Final = _logger(logger)
    details, replaced = _integration(callback).logging_hook(
        kwargs=view.model_call_details, result=result, call_type=view.call_type
    )
    view.model_call_details = details
    return replaced


async def async_logging_hook(logger: Logging, callback: CustomLogger, result: object) -> object:
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.litellm_core_utils import redact_messages

    view: Final = _logger(logger)
    redact: Final = cast(
        LegacyCall, redact_messages.redact_message_input_output_from_custom_logger
    )  # cast-ok: legacy helper
    redacted: Final = (
        result
        if isinstance(callback, CustomGuardrail)
        else redact(result=result, litellm_logging_obj=logger, custom_logger=callback)
    )
    details, replaced = await _integration(callback).async_logging_hook(
        kwargs=view.model_call_details, result=redacted, call_type=view.call_type
    )
    view.model_call_details = details
    return replaced


def mark_logged(logger: Logging, marker: str) -> None:
    _logger(logger).model_call_details[marker] = True


def already_logged(logger: Logging, marker: str) -> bool:
    return _logger(logger).model_call_details.get(marker, False) is True


def log_success_event(
    logger: Logging, callback: CustomLogger, response: object, start_time: Timestamp, end_time: Timestamp
) -> None:
    _integration(callback).log_success_event(
        kwargs=_logger(logger).model_call_details, response_obj=response, start_time=start_time, end_time=end_time
    )


def async_log_success_event(
    logger: Logging, callback: CustomLogger, response: object, start_time: Timestamp, end_time: Timestamp
) -> Awaitable[None]:
    from litellm.litellm_core_utils import redact_messages

    integration: Final = _integration(callback)
    details: Final = integration.redact_standard_logging_payload_from_model_call_details(
        model_call_details=_logger(logger).model_call_details
    )
    redact: Final = cast(
        Callable[..., Details], redact_messages.redact_streaming_responses_for_custom_logger
    )  # cast-ok: legacy helper
    view: Final = redact(model_call_details=details, custom_logger=callback)
    return integration.async_log_success_event(
        kwargs=view, response_obj=response, start_time=start_time, end_time=end_time
    )


def log_failure_event(logger: Logging, callback: CustomLogger, start_time: Timestamp, end_time: Timestamp) -> None:
    _integration(callback).log_failure_event(
        kwargs=_logger(logger).model_call_details, response_obj=None, start_time=start_time, end_time=end_time
    )


def async_log_failure_event(
    logger: Logging, callback: CustomLogger, start_time: Timestamp, end_time: Timestamp
) -> Awaitable[None]:
    return _integration(callback).async_log_failure_event(
        kwargs=_logger(logger).model_call_details, response_obj=None, start_time=start_time, end_time=end_time
    )


def _custom_logger_singleton() -> IntegrationView:
    from litellm.litellm_core_utils import litellm_logging

    existing: Final = _legacy_module().get("customLogger")
    if isinstance(existing, CustomLogger):
        return _integration(existing)
    created: Final = CustomLogger()
    litellm_logging.customLogger = created  # pyright: ignore[reportAttributeAccessIssue]  # legacy module global
    return _integration(created)


def dispatch_callable(
    logger: Logging,
    callback: LegacyCall,
    family: TerminalFamily,
    response: object,
    start_time: Timestamp,
    end_time: Timestamp,
) -> Awaitable[None] | None:
    custom: Final = _custom_logger_singleton()
    details: Final = _logger(logger).model_call_details
    match family:
        case "sync_success" | "sync_failure":
            custom.log_event(
                kwargs=details,
                response_obj=response,
                start_time=start_time,
                end_time=end_time,
                print_verbose=_print_verbose(),
                callback_func=callback,
            )
            return None
        case "async_success" | "async_failure":
            return custom.async_log_event(
                kwargs=details,
                response_obj=response,
                start_time=start_time,
                end_time=end_time,
                print_verbose=_print_verbose(),
                callback_func=callback,
            )


_SUCCESS_SINGLETONS: Final[Mapping[str, str]] = {
    "promptlayer": "promptLayerLogger",
    "supabase": "supabaseClient",
    "wandb": "weightsBiasesLogger",
    "logfire": "logfireLogger",
    "lunary": "lunaryLogger",
    "helicone": "heliconeLogger",
    "greenscale": "greenscaleLogger",
    "athina": "athinaLogger",
    "traceloop": "traceloopLogger",
    "s3": "s3Logger",
    "openmeter": "openMeterLogger",
}


def dispatch_named_success(
    logger: Logging, name: str, response: object, start_time: Timestamp, end_time: Timestamp
) -> Awaitable[None] | None:
    view: Final = _logger(logger)
    details: Final = view.model_call_details
    print_verbose: Final = _print_verbose()
    integration: Final = _legacy_module().get(_SUCCESS_SINGLETONS.get(name, ""))
    without_response: Final = {key: value for key, value in details.items() if key != "original_response"}
    match name:
        case "promptlayer" | "wandb" | "athina" if integration is not None:
            _method(integration, "log_event")(
                kwargs=details,
                response_obj=response,
                start_time=start_time,
                end_time=end_time,
                print_verbose=print_verbose,
            )
        case "logfire" if integration is not None:
            from litellm.integrations.logfire_logger import LogfireLevel

            _method(integration, "log_event")(
                kwargs=without_response,
                response_obj=response,
                start_time=start_time,
                end_time=end_time,
                print_verbose=print_verbose,
                level=LogfireLevel.INFO.value,
            )
        case "greenscale" if integration is not None:
            _method(integration, "log_event")(
                kwargs=without_response,
                response_obj=response,
                start_time=start_time,
                end_time=end_time,
                print_verbose=print_verbose,
            )
        case "supabase" if integration is not None:
            _method(integration, "log_event")(
                model=view.model,
                messages=view.messages,
                end_user=details.get("user", "default"),
                response_obj=response,
                start_time=start_time,
                end_time=end_time,
                litellm_call_id=details["litellm_call_id"],
                print_verbose=print_verbose,
            )
        case "lunary" if integration is not None:
            _method(integration, "log_event")(
                kwargs=details,
                type="llm",
                event="end",
                model=view.model,
                input=details["input"],
                user_id=details.get("user", "default"),
                response_obj=response,
                start_time=start_time,
                end_time=end_time,
                run_id=view.litellm_call_id,
                print_verbose=print_verbose,
            )
        case "helicone" if integration is not None:
            _method(integration, "log_success")(
                model=view.model,
                messages=view.messages,
                response_obj=response,
                start_time=start_time,
                end_time=end_time,
                print_verbose=print_verbose,
                kwargs=details,
            )
        case "langfuse":
            _langfuse(
                logger, response=response, start_time=start_time, end_time=end_time, level=None, status_message=None
            )
        case "traceloop" if integration is not None:
            _method(integration, "log_event")(
                kwargs=details,
                response_obj=response,
                start_time=start_time,
                end_time=end_time,
                user_id=details.get("user", None),
                print_verbose=print_verbose,
            )
        case "s3" if integration is not None:
            _method(integration, "log_event")(
                kwargs=details,
                response_obj=response,
                start_time=start_time,
                end_time=end_time,
                print_verbose=print_verbose,
            )
        case "openmeter" if integration is not None:
            return _async_method(integration, "async_log_success_event")(
                kwargs=details, response_obj=response, start_time=start_time, end_time=end_time
            )
        case "dynamodb":
            return _dynamodb(details, response, start_time, end_time, print_verbose)
        case _:
            return None
    return None


def _dynamodb(
    details: Details, response: object, start_time: Timestamp, end_time: Timestamp, print_verbose: LegacyCall
) -> Awaitable[None]:
    from litellm.integrations.dynamodb import DyanmoDBLogger
    from litellm.litellm_core_utils import litellm_logging

    existing: Final = _legacy_module().get("dynamoLogger")
    dynamo: Final = existing if isinstance(existing, DyanmoDBLogger) else DyanmoDBLogger()
    litellm_logging.dynamoLogger = dynamo  # pyright: ignore[reportAttributeAccessIssue]  # legacy module global
    return _async_method(dynamo, "_async_log_event")(
        kwargs=details, response_obj=response, start_time=start_time, end_time=end_time, print_verbose=print_verbose
    )


def _langfuse(
    logger: Logging,
    *,
    response: object,
    start_time: Timestamp,
    end_time: Timestamp,
    level: str | None,
    status_message: str | None,
) -> None:
    from litellm.integrations.langfuse import langfuse_handler

    view: Final = _logger(logger)
    module: Final = _legacy_module()
    kwargs: Final = {key: value for key, value in view.model_call_details.items() if key != "original_response"}
    select: Final = cast(
        LegacyCall, langfuse_handler.LangFuseHandler.get_langfuse_logger_for_request
    )  # cast-ok: legacy factory
    handler: Final = select(
        globalLangfuseLogger=module.get("langFuseLogger"),
        standard_callback_dynamic_params=view.standard_callback_dynamic_params,
        in_memory_dynamic_logger_cache=module["in_memory_dynamic_logger_cache"],
    )
    if handler is None:
        return
    extra: Final[Details] = {"level": level, "status_message": status_message} if level is not None else {}
    result: Final = _method(handler, "log_event_on_langfuse")(
        kwargs=kwargs,
        response_obj=response,
        start_time=start_time,
        end_time=end_time,
        user_id=kwargs.get("user", None),
        **extra,
    )
    trace_id: Final = (
        cast(Details, result).get("trace_id") if isinstance(result, dict) else None
    )  # cast-ok: legacy response dict
    if trace_id is not None:
        _method(module["in_memory_trace_id_cache"], "set_cache")(
            litellm_call_id=view.litellm_call_id, service_name="langfuse", trace_id=trace_id
        )


def dispatch_named_failure(
    logger: Logging,
    name: str,
    exception: BaseException,
    formatted: str,
    start_time: Timestamp,
    end_time: Timestamp,
) -> None:
    view: Final = _logger(logger)
    module: Final = _legacy_module()
    details: Final = view.model_call_details
    print_verbose: Final = _print_verbose()
    match name:
        case "lunary" if (lunary := module.get("lunaryLogger")) is not None:
            _method(lunary, "log_event")(
                kwargs=details,
                type="llm",
                event="error",
                user_id=details.get("user", "default"),
                model=view.model,
                input=details["input"],
                error=formatted,
                run_id=view.litellm_call_id,
                start_time=start_time,
                end_time=end_time,
                print_verbose=print_verbose,
            )
        case "sentry" if (capture := module.get("capture_exception")) is not None:
            cast(LegacyCall, capture)(exception)  # cast-ok: legacy sentry hook
        case "supabase" if (supabase := module.get("supabaseClient")) is not None:
            _method(supabase, "log_event")(
                model=view.model,
                messages=view.messages,
                end_user=details.get("user", "default"),
                response_obj=None,
                start_time=start_time,
                end_time=end_time,
                litellm_call_id=details["litellm_call_id"],
                print_verbose=print_verbose,
            )
        case "langfuse":
            _langfuse(
                logger,
                response=None,
                start_time=start_time,
                end_time=end_time,
                level="ERROR",
                status_message=str(exception),
            )
        case "traceloop" if (traceloop := module.get("traceloopLogger")) is not None:
            _method(traceloop, "log_event")(
                start_time=start_time,
                end_time=end_time,
                response_obj=None,
                user_id=details.get("user", None),
                print_verbose=print_verbose,
                status_message=str(exception),
                level="ERROR",
                kwargs=details,
            )
        case "logfire" if (logfire := module.get("logfireLogger")) is not None:
            from litellm.integrations.logfire_logger import LogfireLevel

            _method(logfire, "log_event")(
                kwargs={
                    **{key: value for key, value in details.items() if key != "original_response"},
                    "exception": exception,
                },
                response_obj=None,
                start_time=start_time,
                end_time=end_time,
                level=LogfireLevel.ERROR.value,
                print_verbose=print_verbose,
            )
        case _:
            return


def restore_correlation_context(logger: object) -> None:
    from litellm import utils

    utils._restore_correlation_context_if_supported(logger)  # pyright: ignore[reportPrivateUsage]  # legacy interop helper


def submit_worker(job: Callable[[], None]) -> None:
    import contextvars

    from litellm.litellm_core_utils.thread_pool_executor import executor

    context: Final = contextvars.copy_context()
    _ = executor.submit(context.run, job)


def enqueue_background(coroutine: Awaitable[None]) -> None:
    import contextvars

    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

    enqueue: Final = cast(
        Callable[[Awaitable[None]], None], GLOBAL_LOGGING_WORKER.ensure_initialized_and_enqueue
    )  # cast-ok: legacy worker accepts any coroutine
    contextvars.copy_context().run(enqueue, coroutine)


def now() -> Timestamp:
    return datetime.datetime.now()
