"""
This integration ingests tracing data from native extensions written in Rust.

Using it requires additional setup on the Rust side to accept a
`RustTracingLayer` Python object and register it with the `tracing-subscriber`
using an adapter from the `pyo3-python-tracing-subscriber` crate. For example:
```rust
#[pyfunction]
pub fn initialize_tracing(py_impl: Bound<'_, PyAny>) {
    tracing_subscriber::registry()
        .with(pyo3_python_tracing_subscriber::PythonCallbackLayerBridge::new(py_impl))
        .init();
}
```

Usage in Python would then look like:
```
sentry_sdk.init(
    dsn=sentry_dsn,
    integrations=[
        RustTracingIntegration(
            "demo_rust_extension",
            demo_rust_extension.initialize_tracing,
            event_type_mapping=event_type_mapping,
        )
    ],
)
```

Each native extension requires its own integration.
"""

import json
from enum import Enum, auto
from typing import Any, Callable, Dict, Tuple, Optional

import sentry_sdk
from sentry_sdk.integrations import Integration
from sentry_sdk.scope import should_send_default_pii
from sentry_sdk.tracing import Span as SentrySpan
from sentry_sdk.utils import SENSITIVE_DATA_SUBSTITUTE

TraceState = Optional[Tuple[Optional[SentrySpan], SentrySpan]]


class RustTracingLevel(Enum):
    Trace = "TRACE"
    Debug = "DEBUG"
    Info = "INFO"
    Warn = "WARN"
    Error = "ERROR"


class EventTypeMapping(Enum):
    Ignore = auto()
    Exc = auto()
    Breadcrumb = auto()
    Event = auto()


def tracing_level_to_sentry_level(level):
    # type: (str) -> sentry_sdk._types.LogLevelStr
    level = RustTracingLevel(level)
    if level in (RustTracingLevel.Trace, RustTracingLevel.Debug):
        return "debug"
    elif level == RustTracingLevel.Info:
        return "info"
    elif level == RustTracingLevel.Warn:
        return "warning"
    elif level == RustTracingLevel.Error:
        return "error"
    else:
        # Better this than crashing
        return "info"


def extract_contexts(event: Dict[str, Any]) -> Dict[str, Any]:
    metadata = event.get("metadata", {})
    contexts = {}

    location = {}
    for field in ["module_path", "file", "line"]:
        if field in metadata:
            location[field] = metadata[field]
    if len(location) > 0:
        contexts["rust_tracing_location"] = location

    fields = {}
    for field in metadata.get("fields", []):
        fields[field] = event.get(field)
    if len(fields) > 0:
        contexts["rust_tracing_fields"] = fields

    return contexts


def process_event(event: Dict[str, Any]) -> None:
    metadata = event.get("metadata", {})

    logger = metadata.get("target")
    level = tracing_level_to_sentry_level(metadata.get("level"))
    message = event.get("message")  # type: sentry_sdk._types.Any
    contexts = extract_contexts(event)

    sentry_event = {
        "logger": logger,
        "level": level,
        "message": message,
        "contexts": contexts,
    }  # type: sentry_sdk._types.Event

    sentry_sdk.capture_event(sentry_event)


def process_exception(event: Dict[str, Any]) -> None:
    process_event(event)


def process_breadcrumb(event: Dict[str, Any]) -> None:
    level = tracing_level_to_sentry_level(event.get("metadata", {}).get("level"))
    message = event.get("message")

    sentry_sdk.add_breadcrumb(level=level, message=message)


def default_span_filter(metadata: Dict[str, Any]) -> bool:
    return RustTracingLevel(metadata.get("level")) in (
        RustTracingLevel.Error,
        RustTracingLevel.Warn,
        RustTracingLevel.Info,
    )


def default_event_type_mapping(metadata: Dict[str, Any]) -> EventTypeMapping:
    level = RustTracingLevel(metadata.get("level"))
    if level == RustTracingLevel.Error:
        return EventTypeMapping.Exc
    elif level in (RustTracingLevel.Warn, RustTracingLevel.Info):
        return EventTypeMapping.Breadcrumb
    elif level in (RustTracingLevel.Debug, RustTracingLevel.Trace):
        return EventTypeMapping.Ignore
    else:
        return EventTypeMapping.Ignore


class RustTracingLayer:
    def __init__(
        self,
        origin: str,
        event_type_mapping: Callable[
            [Dict[str, Any]], EventTypeMapping
        ] = default_event_type_mapping,
        span_filter: Callable[[Dict[str, Any]], bool] = default_span_filter,
        include_tracing_fields: Optional[bool] = None,
    ):
        self.origin = origin
        self.event_type_mapping = event_type_mapping
        self.span_filter = span_filter
        self.include_tracing_fields = include_tracing_fields

    def _include_tracing_fields(self) -> bool:
        """
        By default, the values of tracing fields are not included in case they
        contain PII. A user may override that by passing `True` for the
        `include_tracing_fields` keyword argument of this integration or by
        setting `send_default_pii` to `True` in their Sentry client options.
        """
        return (
            should_send_default_pii()
            if self.include_tracing_fields is None
            else self.include_tracing_fields
        )

    def on_event(self, event: str, _span_state: TraceState) -> None:
        deserialized_event = json.loads(event)
        metadata = deserialized_event.get("metadata", {})

        event_type = self.event_type_mapping(metadata)
        if event_type == EventTypeMapping.Ignore:
            return
        elif event_type == EventTypeMapping.Exc:
            process_exception(deserialized_event)
        elif event_type == EventTypeMapping.Breadcrumb:
            process_breadcrumb(deserialized_event)
        elif event_type == EventTypeMapping.Event:
            process_event(deserialized_event)

    def on_new_span(self, attrs: str, span_id: str) -> TraceState:
        attrs = json.loads(attrs)
        metadata = attrs.get("metadata", {})

        if not self.span_filter(metadata):
            return None

        module_path = metadata.get("module_path")
        name = metadata.get("name")
        message = attrs.get("message")

        if message is not None:
            sentry_span_name = message
        elif module_path is not None and name is not None:
            sentry_span_name = f"{module_path}::{name}"  # noqa: E231
        elif name is not None:
            sentry_span_name = name
        else:
            sentry_span_name = "<unknown>"

        kwargs = {
            "op": "function",
            "name": sentry_span_name,
            "origin": self.origin,
        }

        scope = sentry_sdk.get_current_scope()
        parent_sentry_span = scope.span
        if parent_sentry_span:
            sentry_span = parent_sentry_span.start_child(**kwargs)
        else:
            sentry_span = scope.start_span(**kwargs)

        fields = metadata.get("fields", [])
        for field in fields:
            if self._include_tracing_fields():
                sentry_span.set_data(field, attrs.get(field))
            else:
                sentry_span.set_data(field, SENSITIVE_DATA_SUBSTITUTE)

        scope.span = sentry_span
        return (parent_sentry_span, sentry_span)

    def on_close(self, span_id: str, span_state: TraceState) -> None:
        if span_state is None:
            return

        parent_sentry_span, sentry_span = span_state
        sentry_span.finish()
        sentry_sdk.get_current_scope().span = parent_sentry_span

    def on_record(self, span_id: str, values: str, span_state: TraceState) -> None:
        if span_state is None:
            return
        _parent_sentry_span, sentry_span = span_state

        deserialized_values = json.loads(values)
        for key, value in deserialized_values.items():
            if self._include_tracing_fields():
                sentry_span.set_data(key, value)
            else:
                sentry_span.set_data(key, SENSITIVE_DATA_SUBSTITUTE)


class RustTracingIntegration(Integration):
    """
    Ingests tracing data from a Rust native extension's `tracing` instrumentation.

    If a project uses more than one Rust native extension, each one will need
    its own instance of `RustTracingIntegration` with an initializer function
    specific to that extension.

    Since all of the setup for this integration requires instance-specific state
    which is not available in `setup_once()`, setup instead happens in `__init__()`.
    """

    def __init__(
        self,
        identifier: str,
        initializer: Callable[[RustTracingLayer], None],
        event_type_mapping: Callable[
            [Dict[str, Any]], EventTypeMapping
        ] = default_event_type_mapping,
        span_filter: Callable[[Dict[str, Any]], bool] = default_span_filter,
        include_tracing_fields: Optional[bool] = None,
    ):
        self.identifier = identifier
        origin = f"auto.function.rust_tracing.{identifier}"
        self.tracing_layer = RustTracingLayer(
            origin, event_type_mapping, span_filter, include_tracing_fields
        )

        initializer(self.tracing_layer)

    @staticmethod
    def setup_once() -> None:
        pass
