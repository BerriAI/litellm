from collections.abc import AsyncGenerator, AsyncIterator, Callable, Mapping
from typing import TYPE_CHECKING, Final

from opentelemetry.sdk.trace import ReadableSpan
from pydantic import TypeAdapter

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.otel.logger import OpenTelemetryV2
from litellm.integrations.otel.mappers.base import AttributeMap
from litellm.integrations.otel.mappers.utils import drop_none_pairs
from litellm.integrations.otel.model.request_io import request_input, response_output, stream_output
from litellm.integrations.otel.plumbing.context import request_root_span
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.redact_messages import (
    should_redact_message_logging,  # pyright: ignore[reportUnknownVariableType]  # the helper types its argument as a bare dict
)

if TYPE_CHECKING:
    from opentelemetry.trace import Span

    from litellm.caching.dual_cache import DualCache
    from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth
    from litellm.types.utils import CallTypesLiteral, ModelResponseStream

_JSON_MIME_TYPE: Final = "application/json"
_LITELLM_PARAMS: Final = TypeAdapter(dict[str, object])


class RootIOOpenTelemetryV2(OpenTelemetryV2):
    """Stamps the request's input and output on the proxy's request root span while it is still recording.

    The root span ends when the response is sent, before the success callback runs, so both stamps come from the
    post-call hooks in the request task: the request as it stands after the pre-call chain and the response as it
    is returned, for the call types whose response renders as a message. Subclasses name the attributes and define
    ``async_post_call_streaming_iterator_hook`` on themselves over ``_relay_stream``, since the proxy relays a stream
    only through a callback whose own class defines that hook.
    """

    relays_served_stream: bool = True

    async def async_post_call_success_hook(
        self,
        data: Mapping[str, object],
        user_api_key_dict: "UserAPIKeyAuth",
        response: object,
    ) -> None:
        self._stamp_root_io(data, lambda: response_output(response))

    async def _relay_stream(
        self,
        response: "AsyncIterator[ModelResponseStream]",
        request_data: Mapping[str, object],
    ) -> "AsyncGenerator[ModelResponseStream, None]":
        if not self._capture_span_content():
            async for chunk in response:
                yield chunk
            return
        relayed: Final[list[ModelResponseStream]] = []  # mutable-ok: relayed as they arrive, assembled at end of stream
        async for chunk in response:
            relayed.append(chunk)
            yield chunk
        self._stamp_root_io(request_data, lambda: stream_output(tuple(relayed), request_data))

    def _stamp_root_io(self, data: Mapping[str, object], render_output: Callable[[], str | None]) -> None:
        if not self._capture_span_content():
            return
        root: Final = request_root_span()
        if root is None or not root.is_recording() or not self._stamps_root(root):
            return
        try:
            output: Final = render_output()
            if output is None:
                return
            rendered_input: Final = request_input(data)
        except Exception:  # noqa: BLE001  # telemetry must never fail the request it describes
            verbose_logger.debug("otel v2: could not render the request root input or output", exc_info=True)
            return
        root.set_attributes(self._root_io_attributes(rendered_input, output))

    def _stamps_root(self, root: "Span") -> bool:
        """Whether this logger writes the input and output onto ``root``."""
        return True

    def _root_io_attributes(self, rendered_input: str | None, output: str) -> AttributeMap:
        raise NotImplementedError


def _withholds_messages(logger: OpenTelemetryV2, data: Mapping[str, object]) -> bool:
    """Whether the request's messages stay off ``logger``'s LLM span, because the request is not logged to it or its
    logged messages are redacted."""
    if logger.message_logging is not True or logger.turn_off_message_logging:
        return True
    logging_obj: Final = data.get("litellm_logging_obj")
    if not isinstance(logging_obj, Logging):
        return (
            data.get("no-log") is True and not litellm.global_disable_no_log_param
        ) or should_redact_message_logging({"litellm_params": dict(data)})
    details: Final[Mapping[str, object]] = logging_obj.model_call_details
    runs_for_logger: Final = logging_obj.should_run_callback(  # pyright: ignore[reportUnknownMemberType]  # Logging types the hook's litellm_params as a bare dict
        logger, _LITELLM_PARAMS.validate_python(details.get("litellm_params", {})), "post_call_hook"
    )
    return not runs_for_logger or should_redact_message_logging(logging_obj.model_call_details)


class OpenInferenceRootIOOpenTelemetryV2(RootIOOpenTelemetryV2):
    """Stamps the request root's OpenInference kind, input and output, which Arize AX and Phoenix show as the
    trace's, on a root this logger's provider started, unless the request's messages stay off this logger's LLM span.

    The proxy runs a callback's pre-call hook only when the callback's own class defines it, so this class repeats
    the inherited one that seeds the request's metadata baggage.
    """

    async def async_pre_call_hook(
        self,
        user_api_key_dict: "UserAPIKeyAuth",
        cache: "DualCache",
        data: dict[str, object],  # mutable-ok: the proxy's pre-call contract hands over the request body
        call_type: "CallTypesLiteral",
    ) -> dict[str, object]:  # mutable-ok: the proxy's pre-call contract returns the request body
        return await super().async_pre_call_hook(user_api_key_dict, cache, data, call_type)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # the inherited hook types its body as a bare dict

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: "UserAPIKeyAuth",
        response: "AsyncIterator[ModelResponseStream]",
        request_data: Mapping[str, object],
    ) -> "AsyncGenerator[ModelResponseStream, None]":
        async for chunk in self._relay_stream(response, request_data):
            yield chunk

    def _stamp_root_io(self, data: Mapping[str, object], render_output: Callable[[], str | None]) -> None:
        if not _withholds_messages(self, data):
            super()._stamp_root_io(data, render_output)

    def _stamps_root(self, root: "Span") -> bool:
        return isinstance(root, ReadableSpan) and root.resource is self.tracer_provider.resource

    def _root_io_attributes(self, rendered_input: str | None, output: str) -> AttributeMap:
        return drop_none_pairs(
            (
                ("openinference.span.kind", "CHAIN"),
                ("input.value", rendered_input),
                ("input.mime_type", _JSON_MIME_TYPE if rendered_input is not None else None),
                ("output.value", output),
                ("output.mime_type", _JSON_MIME_TYPE),
            )
        )
