import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, ClassVar, Final, Literal, TypeAlias

from litellm._logging import verbose_logger
from litellm.proxy._types import UserAPIKeyAuth

if TYPE_CHECKING:
    from prometheus_client import Counter as PrometheusCounter
    from prometheus_client import Gauge as PrometheusGauge
    from prometheus_client import Histogram as PrometheusHistogram

EventCallType: TypeAlias = Literal["mcp_event_subscribe", "mcp_event_poll", "mcp_event_stream", "mcp_event_delivery"]
EventOutcome: TypeAlias = Literal[
    "delivered", "blocked", "throttled", "failed", "terminated", "subscribed", "unsubscribed"
]
DeliveryModeLabel: TypeAlias = Literal["poll", "push", "webhook"]


@dataclass(frozen=True, slots=True)
class EventActivity:
    call_type: EventCallType
    mode: DeliveryModeLabel
    server_id: str
    event_name: str
    outcome: EventOutcome
    caller: UserAPIKeyAuth | None
    subscription_id: str | None = None
    event_ids: tuple[str, ...] = ()
    attempts: int | None = None
    count: int | None = None


class EventMetrics:
    _initialized: ClassVar[bool] = False
    events: ClassVar["PrometheusCounter | None"] = None
    latency: ClassVar["PrometheusHistogram | None"] = None
    subscriptions: ClassVar["PrometheusGauge | None"] = None

    @classmethod
    def _ensure(cls) -> None:
        if cls._initialized:
            return
        cls._initialized = True
        try:
            from prometheus_client import Counter, Gauge, Histogram
        except ImportError:
            return
        cls.events = Counter(
            "litellm_mcp_events_total", "MCP events relayed by the gateway", ("server", "event", "outcome")
        )
        cls.latency = Histogram(
            "litellm_mcp_event_delivery_latency_seconds", "Time from upstream receipt to client acknowledgement"
        )
        cls.subscriptions = Gauge("litellm_mcp_event_subscriptions", "MCP event webhook subscriptions", ("status",))

    @classmethod
    def count(cls, server_id: str, event_name: str, outcome: EventOutcome, amount: int = 1) -> None:
        cls._ensure()
        if cls.events is not None and amount > 0:
            cls.events.labels(server=server_id, event=event_name, outcome=outcome).inc(amount)

    @classmethod
    def observe_latency(cls, seconds: float) -> None:
        cls._ensure()
        if cls.latency is not None:
            cls.latency.observe(seconds)

    @classmethod
    def set_subscriptions(cls, status: str, value: int) -> None:
        cls._ensure()
        if cls.subscriptions is not None:
            cls.subscriptions.labels(status=status).set(value)


LogWriter: TypeAlias = Callable[[EventActivity, UserAPIKeyAuth], Awaitable[None]]


async def record(activity: EventActivity, write: LogWriter | None = None) -> None:
    amount: Final = activity.count if activity.count is not None else max(len(activity.event_ids), 1)
    EventMetrics.count(activity.server_id, activity.event_name, activity.outcome, amount)
    if activity.caller is None:
        return
    try:
        await (write or write_log_row)(activity, activity.caller)
    except Exception as error:  # noqa: BLE001  # logging must never change what the client receives
        verbose_logger.warning("Failed to log MCP event activity: %s", error)


async def write_log_row(activity: EventActivity, caller: UserAPIKeyAuth) -> None:
    """One spend-log row per event outcome, attributed to the caller, so event traffic shows next to tool calls."""
    from mcp.types import CallToolResult, TextContent

    from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup
    from litellm.types.utils import CallTypes
    from litellm.utils import Rules, function_setup

    start: Final = datetime.now()  # noqa: DTZ005  # spend logs use naive timestamps
    details: Final = {
        "call_type": activity.call_type,
        "mode": activity.mode,
        "server_id": activity.server_id,
        "event_name": activity.event_name,
        "outcome": activity.outcome,
        "subscription_id": activity.subscription_id,
        "event_ids": list(activity.event_ids),
        "attempts": activity.attempts,
        "count": activity.count,
    }
    metadata: Final = {
        **LiteLLMProxyRequestSetup.get_sanitized_user_information_from_key(caller),
        "spend_logs_metadata": {"mcp_event": details},
    }
    logging_obj, _request = function_setup(
        original_function=CallTypes.call_mcp_tool.value,
        rules_obj=Rules(),
        start_time=start,
        model=f"MCP: {activity.call_type}",
        litellm_call_id=str(uuid.uuid4()),
        metadata=metadata,
    )
    logging_obj.update_environment_variables(  # pyright: ignore[reportUnknownMemberType]  # untyped legacy signature
        litellm_params={"metadata": metadata}, optional_params={}, user=caller.user_id
    )
    summary: Final = CallToolResult(content=[TextContent(type="text", text=f"{activity.call_type} {activity.outcome}")])
    await logging_obj.async_success_handler(result=summary, start_time=start, end_time=datetime.now())  # noqa: DTZ005  # spend logs use naive timestamps
