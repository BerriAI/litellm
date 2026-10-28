import base64
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from itertools import chain
from typing import TYPE_CHECKING, Any, Final, Optional, Protocol, runtime_checkable

from opentelemetry import trace

import litellm
from litellm._logging import verbose_logger
from litellm.constants import REDACTED_BY_LITELLM
from litellm.integrations._types.open_inference import ErrorAttributes
from litellm.integrations.arize import _utils
from litellm.integrations.langfuse.langfuse_otel_attributes import (
    LangfuseLLMObsOTELAttributes,
    serialize_langfuse_datetime,
)
from litellm.integrations.opentelemetry import (
    HTTP_RESPONSE_STATUS_CODE_ATTRIBUTE,
    LITELLM_PROXY_REQUEST_SPAN_NAME,
    OpenTelemetry,
    OpenTelemetryConfig,
)
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.litellm_core_utils.safe_json_loads import safe_json_loads
from litellm.secret_managers.main import str_to_bool
from litellm.types.integrations.langfuse_otel import (
    LangfuseSpanAttributes,
)
from litellm.types.utils import StandardCallbackDynamicParams

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    Span = _Span
else:
    Span = Any


LANGFUSE_CLOUD_EU_ENDPOINT: Final = "https://cloud.langfuse.com/api/public/otel"
LANGFUSE_CLOUD_US_ENDPOINT: Final = "https://us.cloud.langfuse.com/api/public/otel"
LANGFUSE_INGESTION_VERSION_HEADER: Final = "x-langfuse-ingestion-version"
LANGFUSE_INGESTION_VERSION: Final = "4"
_ERROR_ATTRIBUTES: Final = frozenset(
    {
        ErrorAttributes.ERROR_CODE,
        HTTP_RESPONSE_STATUS_CODE_ATTRIBUTE,
        ErrorAttributes.ERROR_TYPE,
        ErrorAttributes.ERROR_MESSAGE,
        ErrorAttributes.ERROR_LLM_PROVIDER,
        ErrorAttributes.ERROR_STACK_TRACE,
    }
)


@runtime_checkable
class _Gettable(Protocol):
    def get(self, key: str, default: object = None) -> object: ...


def _as_gettable(value: object) -> _Gettable | None:
    return value if isinstance(value, _Gettable) else None


def _attr(item: object, name: str, default: object = None) -> object:
    return getattr(item, name, default)


def _preset_cache_key(kwargs: Mapping[str, object]) -> object:
    if litellm.cache is None:
        return None
    return litellm.cache.get_preset_cache_key_from_kwargs(**kwargs)


def _mask_flag(value: object) -> bool:
    parsed: Final = str_to_bool(value) if isinstance(value, str) else None
    return bool(value) if parsed is None else parsed


def _caller_tag_list(value: object) -> tuple[str, ...]:
    parsed: Final = (
        safe_json_loads(value, default=value) if isinstance(value, str) and value.strip().startswith("[") else value
    )
    if isinstance(parsed, list):
        return tuple(tag if isinstance(tag, str) else json.dumps(tag, default=str) for tag in parsed)
    if isinstance(parsed, str) and parsed:
        return tuple(tag.strip() for tag in parsed.split(",")) if "," in parsed else (parsed,)
    return ()


def _parent_of(span: object) -> trace.SpanContext | None:
    parent: Final = getattr(span, "parent", None)
    return parent if isinstance(parent, trace.SpanContext) else None


def _litellm_proxy_span(kwargs: Mapping[str, object]) -> trace.Span | None:
    litellm_params: Final = _as_gettable(kwargs.get("litellm_params")) or {}
    carried: Final = (
        (_as_gettable(litellm_params.get(key)) or {}).get("litellm_parent_otel_span")
        for key in ("metadata", "litellm_metadata")
    )
    return next(
        (
            span
            for span in carried
            if isinstance(span, trace.Span) and getattr(span, "name", None) == LITELLM_PROXY_REQUEST_SPAN_NAME
        ),
        None,
    )


def _starts_its_own_trace(span: object, kwargs: Mapping[str, object]) -> bool:
    if span is trace.get_current_span():
        return False
    parent: Final = _parent_of(span)
    if parent is None:
        return True
    proxy_span: Final = _litellm_proxy_span(kwargs)
    return (
        proxy_span is not None
        and _parent_of(proxy_span) is None
        and proxy_span.get_span_context().span_id == parent.span_id
    )


class LangfuseOtelLogger(OpenTelemetry):
    def __init__(self, config=None, *args, **kwargs):
        # Prevent LangfuseOtelLogger from modifying global environment variables by constructing config manually
        # and passing it to the parent OpenTelemetry class
        if config is None:
            config = self._create_open_telemetry_config_from_langfuse_env()
        super().__init__(config=config, *args, **kwargs)

    @staticmethod
    def set_langfuse_otel_attributes(span: Span, kwargs, response_obj):
        """
        Sets OpenTelemetry span attributes for Langfuse observability.
        Uses the same attribute setting logic as Arize Phoenix for consistency.
        """

        _utils.set_attributes(span, kwargs, response_obj, LangfuseLLMObsOTELAttributes)
        span.set_attribute("langfuse.observation.type", "generation")

        #########################################################
        # Set Langfuse specific attributes
        #########################################################
        LangfuseOtelLogger._set_langfuse_specific_attributes(span=span, kwargs=kwargs, response_obj=response_obj)

    @staticmethod
    def _extract_langfuse_metadata(kwargs: Mapping[str, Any]) -> dict:
        """
        Extracts Langfuse metadata from the standard LiteLLM kwargs structure.

        1. Reads kwargs["litellm_params"]["metadata"] if present and is a dict.
        2. Enriches it with any `langfuse_*` request-header params via the
           existing LangFuseLogger.add_metadata_from_header helper so that proxy
           users get identical behaviour across vanilla and OTEL integrations.
        """
        litellm_params: Final = kwargs.get("litellm_params", {}) or {}
        metadata = litellm_params.get("metadata") or {}
        # Ensure we only work with dicts
        if metadata is None or not isinstance(metadata, dict):
            metadata = {}

        # Re-use header extraction logic from the vanilla logger if available
        try:
            from litellm.integrations.langfuse.langfuse import (
                LangFuseLogger as _LFLogger,
            )

            metadata = _LFLogger.add_metadata_from_header(litellm_params, metadata)
        except Exception:
            # Fallback silently if import fails; header enrichment just won't happen
            pass

        return metadata

    @staticmethod
    def _set_metadata_attributes(span: Span, metadata: dict):
        """Helper to set metadata attributes from mapping."""
        from litellm.integrations.arize._utils import safe_set_attribute

        mapping: Final = {
            "generation_name": LangfuseSpanAttributes.GENERATION_NAME,
            "generation_id": LangfuseSpanAttributes.GENERATION_ID,
            "parent_observation_id": LangfuseSpanAttributes.PARENT_OBSERVATION_ID,
            "mask_input": LangfuseSpanAttributes.MASK_INPUT,
            "mask_output": LangfuseSpanAttributes.MASK_OUTPUT,
            "trace_user_id": LangfuseSpanAttributes.TRACE_USER_ID,
            "session_id": LangfuseSpanAttributes.SESSION_ID,
            "tags": LangfuseSpanAttributes.TAGS,
            "trace_name": LangfuseSpanAttributes.TRACE_NAME,
            "trace_id": LangfuseSpanAttributes.TRACE_ID,
            "trace_metadata": LangfuseSpanAttributes.TRACE_METADATA,
            "trace_release": LangfuseSpanAttributes.RELEASE,
            "existing_trace_id": LangfuseSpanAttributes.EXISTING_TRACE_ID,
            "update_trace_keys": LangfuseSpanAttributes.UPDATE_TRACE_KEYS,
            "debug_langfuse": LangfuseSpanAttributes.DEBUG_LANGFUSE,
        }

        version: Final = (
            metadata.get("trace_version") if metadata.get("trace_version") is not None else metadata.get("version")
        )
        if version is not None:
            safe_set_attribute(span, LangfuseSpanAttributes.VERSION.value, version)

        for key, enum_attr in mapping.items():
            if key in metadata and metadata[key] is not None:
                value = metadata[key]
                if key == "trace_id" and isinstance(value, str):
                    # trace_id must be 32 hex char no dashes for langfuse : Litellm sends uuid with dashes (might be breaking at some point)
                    value = value.replace("-", "")

                if isinstance(value, (list, dict)):
                    try:
                        value = json.dumps(value)
                    except Exception:
                        value = str(value)
                safe_set_attribute(span, enum_attr.value, value)

    @staticmethod
    def _observation_output(response_obj: _Gettable | None) -> str | None:
        if not response_obj or not hasattr(response_obj, "get"):
            return None
        return _extract_output_items(response_obj) or _extract_choices_output(response_obj)

    @staticmethod
    def _trace_tags(kwargs: Mapping[str, object], metadata: Mapping[str, object]) -> tuple[str, ...]:
        caller_tags: Final = metadata.get("tags")
        request_tags: Final = (_as_gettable(kwargs.get("standard_logging_object")) or {}).get("request_tags")
        default_tags: Final = litellm.langfuse_default_tags

        def _default_tag(key: str) -> str | None:
            if key == "cache_hit":
                return f"cache_hit:{kwargs.get('cache_hit', False)}"
            if key == "cache_key":
                hidden_params: Final = _as_gettable(metadata.get("hidden_params", {})) or {}
                cache_key: Final = (
                    hidden_params.get("cache_key")
                    if hidden_params.get("cache_key") is not None
                    else _preset_cache_key(kwargs)
                )
                return f"cache_key:{cache_key}"
            if key == "proxy_base_url":
                proxy_base_url: Final = os.environ.get("PROXY_BASE_URL")
                return f"proxy_base_url:{proxy_base_url}" if proxy_base_url is not None else None
            if key in metadata and metadata[key] is not None:
                return f"{key}:{metadata[key]}"
            return None

        expanded: Final = tuple(_default_tag(key) for key in default_tags) if isinstance(default_tags, list) else ()
        candidates: Final = (
            _caller_tag_list(caller_tags)
            + (tuple(tag for tag in request_tags if isinstance(tag, str)) if isinstance(request_tags, list) else ())
            + tuple(tag for tag in expanded if tag is not None)
        )
        return tuple(dict.fromkeys(candidates))

    @staticmethod
    def _set_langfuse_specific_attributes(span: Span, kwargs: Mapping[str, object], response_obj):
        """
        Sets Langfuse specific metadata attributes onto the OTEL span.

        All keys supported by the vanilla Langfuse integration are mapped to
        OTEL-safe attribute names defined in LangfuseSpanAttributes.  Complex
        values (lists/dicts) are serialised to JSON strings for OTEL
        compatibility.
        """
        from litellm.integrations.arize._utils import safe_set_attribute
        from litellm.litellm_core_utils.safe_json_dumps import safe_dumps

        dynamic_params: Final = as_str_mapping(kwargs.get("standard_callback_dynamic_params"))
        langfuse_environment: Final = (
            dynamic_params.get("langfuse_environment") if dynamic_params else None
        ) or os.environ.get("LANGFUSE_TRACING_ENVIRONMENT")
        if langfuse_environment:
            safe_set_attribute(
                span,
                LangfuseSpanAttributes.LANGFUSE_ENVIRONMENT.value,
                langfuse_environment,
            )

        metadata: Final = LangfuseOtelLogger._extract_langfuse_metadata(kwargs)
        LangfuseOtelLogger._set_metadata_attributes(span=span, metadata=metadata)

        joins_existing_trace: Final = metadata.get("existing_trace_id") is not None or not _starts_its_own_trace(
            span, kwargs
        )
        input_json: Final = safe_dumps(kwargs.get("messages")) if kwargs.get("messages") else None
        if input_json is not None:
            safe_set_attribute(span, LangfuseSpanAttributes.OBSERVATION_INPUT.value, input_json)

        completion_start_time: Final = kwargs.get("completion_start_time")
        if isinstance(completion_start_time, datetime):
            safe_set_attribute(
                span,
                LangfuseSpanAttributes.OBSERVATION_COMPLETION_START_TIME.value,
                serialize_langfuse_datetime(completion_start_time),
            )

        output_json: Final = LangfuseOtelLogger._observation_output(response_obj)
        if output_json is not None:
            safe_set_attribute(span, LangfuseSpanAttributes.OBSERVATION_OUTPUT.value, output_json)

        if not joins_existing_trace:
            LangfuseOtelLogger._set_trace_fields(
                span=span, kwargs=kwargs, metadata=metadata, input_json=input_json, output_json=output_json
            )

    @staticmethod
    def _set_trace_fields(
        span: Span,
        kwargs: Mapping[str, object],
        metadata: Mapping[str, object],
        input_json: str | None,
        output_json: str | None,
    ) -> None:
        from litellm.integrations.otel.emitter import carried_keys, span_attribute_limit

        tags: Final = LangfuseOtelLogger._trace_tags(kwargs, metadata)
        fields: Final = (
            *(
                ((LangfuseSpanAttributes.TRACE_NAME.value, f"litellm-{kwargs.get('call_type') or 'completion'}"),)
                if metadata.get("trace_name") is None
                else ()
            ),
            *(((LangfuseSpanAttributes.TAGS.value, json.dumps(list(tags))),) if tags else ()),
            *(
                (
                    (
                        LangfuseSpanAttributes.TRACE_INPUT.value,
                        REDACTED_BY_LITELLM if _mask_flag(metadata.get("mask_input")) else input_json,
                    ),
                )
                if input_json is not None
                else ()
            ),
            *(
                (
                    (
                        LangfuseSpanAttributes.TRACE_OUTPUT.value,
                        REDACTED_BY_LITELLM if _mask_flag(metadata.get("mask_output")) else output_json,
                    ),
                )
                if output_json is not None
                else ()
            ),
        )
        limit: Final = span_attribute_limit(span)
        carried: Final = carried_keys(span)
        new_keys: Final = tuple(key for key, _value in fields if key not in carried)
        room: Final = (
            len(new_keys) if limit is None else max(limit - len(carried) - len(_ERROR_ATTRIBUTES - carried), 0)
        )
        written: Final = carried | frozenset(new_keys[:room])
        for key, value in fields:
            if key in written:
                _utils.safe_set_attribute(span, key, value)

    @staticmethod
    def _get_langfuse_otel_host() -> str | None:
        """
        Returns the Langfuse OTEL host based on environment variables.

        Returned in the following order of precedence:
        1. LANGFUSE_OTEL_HOST
        2. LANGFUSE_HOST
        """
        return os.environ.get("LANGFUSE_OTEL_HOST") or os.environ.get("LANGFUSE_HOST")

    def _create_open_telemetry_config_from_langfuse_env(self) -> OpenTelemetryConfig:
        """
        Creates OpenTelemetryConfig from Langfuse environment variables.
        Does NOT modify global environment variables.
        """
        from litellm.integrations.opentelemetry import OpenTelemetryConfig

        public_key: Final = os.environ.get("LANGFUSE_PUBLIC_KEY", None)
        secret_key: Final = os.environ.get("LANGFUSE_SECRET_KEY", None)

        if not public_key or not secret_key:
            # If no keys, return default from env (likely logging to console or something else)
            return OpenTelemetryConfig.from_env()

        return LangfuseOtelLogger._build_langfuse_otel_config(
            public_key=public_key,
            secret_key=secret_key,
            langfuse_host=LangfuseOtelLogger._get_langfuse_otel_host(),
        )

    @staticmethod
    def get_langfuse_otel_config() -> "OpenTelemetryConfig":
        """
        Retrieves the Langfuse OpenTelemetry configuration based on environment variables.

        Environment Variables:
            LANGFUSE_PUBLIC_KEY: Required. Langfuse public key for authentication.
            LANGFUSE_SECRET_KEY: Required. Langfuse secret key for authentication.
            LANGFUSE_HOST: Optional. Custom Langfuse host URL. Defaults to US cloud.

        Returns:
            OpenTelemetryConfig: A Pydantic model containing Langfuse OTEL configuration.

        Raises:
            ValueError: If required keys are missing.
        """
        public_key: Final = os.environ.get("LANGFUSE_PUBLIC_KEY", None)
        secret_key: Final = os.environ.get("LANGFUSE_SECRET_KEY", None)

        if not public_key or not secret_key:
            raise ValueError(
                "LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must be set for Langfuse OpenTelemetry integration."
            )

        return LangfuseOtelLogger._build_langfuse_otel_config(
            public_key=public_key,
            secret_key=secret_key,
            langfuse_host=LangfuseOtelLogger._get_langfuse_otel_host(),
        )

    @staticmethod
    def _build_langfuse_otel_config(
        public_key: str, secret_key: str, langfuse_host: str | None
    ) -> "OpenTelemetryConfig":
        """
        Builds an OTLP HTTP config pointing at the Langfuse OTEL endpoint for the
        given host (US cloud when no host is provided), authorized with the given keys.
        """
        if langfuse_host:
            normalized_host: Final = langfuse_host if langfuse_host.startswith("http") else f"https://{langfuse_host}"
            endpoint = f"{normalized_host.rstrip('/')}/api/public/otel"
            verbose_logger.debug("Using Langfuse OTEL endpoint from host: %s", endpoint)
        else:
            endpoint = LANGFUSE_CLOUD_US_ENDPOINT
            verbose_logger.debug("Using Langfuse US cloud endpoint: %s", endpoint)

        auth_header: Final = LangfuseOtelLogger._get_langfuse_authorization_header(
            public_key=public_key, secret_key=secret_key
        )

        return OpenTelemetryConfig(
            exporter="otlp_http",
            endpoint=endpoint,
            headers=LangfuseOtelLogger._format_otel_headers(
                LangfuseOtelLogger._build_langfuse_otel_headers(auth_header)
            ),
        )

    @staticmethod
    def _get_langfuse_authorization_header(public_key: str, secret_key: str) -> str:
        """
        Get the authorization header for Langfuse OpenTelemetry.
        """
        auth_string: Final = f"{public_key}:{secret_key}"
        auth_header: Final = base64.b64encode(auth_string.encode()).decode()
        return f"Basic {auth_header}"

    @staticmethod
    def _build_langfuse_otel_headers(auth_header: str) -> dict[str, str]:
        """
        Build the OTLP header set Langfuse expects.

        `x-langfuse-ingestion-version: 4` selects Langfuse's v4 ingestion path;
        without it spans fall back to the older transformation path.
        """
        return {
            "Authorization": auth_header,
            LANGFUSE_INGESTION_VERSION_HEADER: LANGFUSE_INGESTION_VERSION,
        }

    @staticmethod
    def _format_otel_headers(headers: dict[str, str]) -> str:
        """
        Serialize a header mapping into the comma-separated OTLP header string
        """
        return ",".join(f"{key}={value}" for key, value in headers.items())

    def construct_dynamic_otel_headers(
        self, standard_callback_dynamic_params: StandardCallbackDynamicParams
    ) -> dict | None:
        """
        Construct dynamic Langfuse headers from standard callback dynamic params

        This is used for team/key based logging.

        Returns:
            dict: A dictionary of dynamic Langfuse headers
        """
        dynamic_headers: Final = {}

        dynamic_langfuse_public_key: Final = standard_callback_dynamic_params.get("langfuse_public_key")
        dynamic_langfuse_secret_key: Final = standard_callback_dynamic_params.get("langfuse_secret_key")
        if dynamic_langfuse_public_key and dynamic_langfuse_secret_key:
            auth_header: Final = LangfuseOtelLogger._get_langfuse_authorization_header(
                public_key=dynamic_langfuse_public_key,
                secret_key=dynamic_langfuse_secret_key,
            )
            dynamic_headers.update(LangfuseOtelLogger._build_langfuse_otel_headers(auth_header))

        return dynamic_headers

    def construct_dynamic_otel_config(
        self, standard_callback_dynamic_params: StandardCallbackDynamicParams
    ) -> Optional["OpenTelemetryConfig"]:
        """
        Build a full per-request OTLP config from team/key dynamic Langfuse credentials.

        Key-scoped credentials must define the export target, not just the auth
        headers: without this, a proxy with no global LANGFUSE_* env vars keeps its
        init-time fallback exporter (console), so key-level langfuse_otel silently
        never reaches Langfuse.
        """
        public_key: Final = standard_callback_dynamic_params.get("langfuse_public_key")
        secret_key: Final = standard_callback_dynamic_params.get("langfuse_secret_key")
        if not public_key or not secret_key:
            return None

        langfuse_host: Final = standard_callback_dynamic_params.get("langfuse_host") or self._get_langfuse_otel_host()
        return LangfuseOtelLogger._build_langfuse_otel_config(
            public_key=public_key,
            secret_key=secret_key,
            langfuse_host=langfuse_host,
        )

    def create_litellm_proxy_request_started_span(
        self,
        start_time: datetime,
        headers: dict,
    ) -> Span | None:
        """
        Override to prevent creating empty proxy request spans.

        Langfuse should only receive spans for actual LLM calls, not for
        internal proxy operations (auth, postgres, proxy_pre_call, etc.).

        By returning None, we prevent the parent span from being created,
        which in turn prevents empty traces from being sent to Langfuse.
        """
        return None

    async def async_service_success_hook(self, *args, **kwargs):
        """
        Langfuse should not receive service success logs.
        """

    async def async_service_failure_hook(self, *args, **kwargs):
        """
        Langfuse should not receive service failure logs.
        """


def _extract_choices_output(response_obj: _Gettable) -> str | None:
    choices: Final = response_obj.get("choices", [])
    if not isinstance(choices, list) or not choices:
        return None
    first_choice: Final = _as_gettable(choices[0])
    if first_choice is None:
        return None
    message: Final = _as_gettable(first_choice.get("message", {}))
    if message is None:
        return None
    tool_calls: Final = message.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        transformed_tool_calls: Final = [
            entry for tool_call in tool_calls if (entry := _transformed_tool_call(response_obj, tool_call)) is not None
        ]
        return safe_dumps(transformed_tool_calls)
    output_data: Final = {
        key: value
        for key, value in (
            ("role", message.get("role") or None),
            ("content", message.get("content")),
        )
        if value is not None
    }
    return safe_dumps(output_data) if output_data else None


def _transformed_tool_call(response_obj: _Gettable, tool_call: object) -> dict[str, object] | None:
    call: Final = _as_gettable(tool_call)
    if call is None:
        return None
    function: Final = _as_gettable(call.get("function", {}))
    if function is None:
        return None
    return {
        "id": response_obj.get("id", ""),
        "name": function.get("name", ""),
        "call_id": call.get("id", ""),
        "type": "function_call",
        "arguments": _tool_call_arguments(function.get("arguments", "{}")),
    }


def _tool_call_arguments(arguments: object) -> object:
    if not isinstance(arguments, str):
        return arguments
    try:
        return json.loads(arguments)
    except json.JSONDecodeError:
        return {}


def _extract_output_items(response_obj: _Gettable) -> str | None:
    output: Final = response_obj.get("output", [])
    if not isinstance(output, list) or not output:
        return None
    rendered: Final = tuple(chain.from_iterable(map(_output_items, output)))
    return safe_dumps(list(rendered)) if rendered else None


def _output_items(item: object) -> tuple[dict[str, object], ...]:
    item_type: Final = _attr(item, "type")
    if item_type == "reasoning":
        summaries: Final = _attr(item, "summary")
        if not isinstance(summaries, Iterable):
            return ()
        return tuple(
            {"role": "reasoning_summary", "content": _attr(summary, "text")}
            for summary in summaries
            if hasattr(summary, "text")
        )
    if item_type == "message":
        content_items: Final = _attr(item, "content", [{}])
        first_content: Final = (
            content_items[0]
            if isinstance(content_items, Sequence) and not isinstance(content_items, (str, bytes)) and content_items
            else {}
        )
        return (
            {
                "role": _attr(item, "role", "assistant"),
                "content": _attr(first_content, "text", ""),
            },
        )
    if item_type == "function_call":
        arguments: Final = _attr(item, "arguments", "{}")
        return (
            {
                "id": _attr(item, "id", ""),
                "name": _attr(item, "name", ""),
                "call_id": _attr(item, "call_id", ""),
                "type": "function_call",
                "arguments": safe_json_loads(arguments, default={}) if isinstance(arguments, str) else arguments,
            },
        )
    return ()
