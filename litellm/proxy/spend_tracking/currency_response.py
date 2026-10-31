import json
from collections.abc import Callable
from typing import Final

from pydantic import TypeAdapter, ValidationError
from starlette.datastructures import MutableHeaders
from starlette.routing import get_route_path
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from litellm.proxy.spend_tracking.currency import Currency

MONEY_ROUTES: Final = (
    "/key/info",
    "/key/list",
    "/team/info",
    "/team/list",
    "/user/info",
    "/user/list",
    "/organization/info",
    "/customer/info",
    "/budget/info",
    "/budget/list",
    "/spend/logs",
    "/user/daily/activity",
    "/team/daily/activity",
    "/organization/daily/activity",
    "/customer/daily/activity",
    "/end_user/daily/activity",
    "/tag/daily/activity",
    "/gateway/daily/activity",
    "/agent/daily/activity/aggregated",
    "/user/daily/activity/aggregated",
    "/team/daily/activity/aggregated",
    "/organization/daily/activity/aggregated",
    "/customer/daily/activity/aggregated",
    "/end_user/daily/activity/aggregated",
    "/tag/daily/activity/aggregated",
    "/user/daily/activity/aggregated/keys",
    "/user/daily/activity/aggregated/search",
    "/user/daily/activity/aggregated/model_top_keys",
    "/user/daily/activity/aggregated/users",
    "/user/daily/activity/aggregated/cache_leakage_keys",
    "/user/daily/activity/export",
    "/team/daily/activity/aggregated/keys",
    "/team/daily/activity/aggregated/search",
    "/team/daily/activity/aggregated/model_top_keys",
    "/team/daily/activity/export",
    "/organization/daily/activity/aggregated/keys",
    "/organization/daily/activity/aggregated/search",
    "/organization/daily/activity/aggregated/model_top_keys",
    "/organization/daily/activity/export",
    "/customer/daily/activity/aggregated/keys",
    "/customer/daily/activity/aggregated/search",
    "/customer/daily/activity/aggregated/model_top_keys",
    "/customer/daily/activity/export",
    "/end_user/daily/activity/aggregated/keys",
    "/end_user/daily/activity/aggregated/search",
    "/end_user/daily/activity/aggregated/model_top_keys",
    "/end_user/daily/activity/export",
    "/tag/daily/activity/aggregated/keys",
    "/tag/daily/activity/aggregated/search",
    "/tag/daily/activity/aggregated/model_top_keys",
    "/tag/daily/activity/export",
    "/agent/daily/activity/aggregated/keys",
    "/agent/daily/activity/aggregated/search",
    "/agent/daily/activity/aggregated/model_top_keys",
    "/agent/daily/activity/export",
    "/global/spend",
    "/global/spend/logs",
    "/global/spend/keys",
    "/global/spend/teams",
    "/global/spend/end_users",
    "/global/spend/models",
    "/global/spend/provider",
    "/global/spend/report",
    "/global/spend/tags",
    "/global/spend/all_tag_names",
)

MONEY_FIELDS: Final = frozenset(
    (
        "spend",
        "max_budget",
        "soft_budget",
        "total_spend",
        "current_spend",
        "budget_limit",
        "cost",
        "max_budget_in_team",
        "team_member_budget",
        "temp_budget_increase",
        "flat_cost",
        "total_flat_cost",
        "compression_savings_spend",
        "prompt_caching_savings_spend",
        "gateway_injected_caching_savings_spend",
        "autorouter_savings_spend",
        "total_compression_savings_spend",
        "total_prompt_caching_savings_spend",
        "total_gateway_injected_caching_savings_spend",
        "total_autorouter_savings_spend",
    )
)
_OBJECT: Final = TypeAdapter(dict[str, object])
_ARRAY: Final = TypeAdapter(list[object])


def _display_values(value: object, rate: float) -> object:
    if isinstance(value, list):
        return [_display_values(item, rate) for item in _ARRAY.validate_python(value)]
    if not isinstance(value, dict):
        return value
    values: Final = _OBJECT.validate_python(value)
    return {
        **{key: _display_values(item, rate) for key, item in values.items()},
        **{
            f"{key}_display": round(item * rate, 10)
            for key, item in values.items()
            if key in MONEY_FIELDS and isinstance(item, (int, float)) and not isinstance(item, bool)
        },
    }


class _CurrencyResponder:
    def __init__(self, send: Send, currency: Currency) -> None:
        self.send = send
        self.currency = currency
        self.start: Message | None = None
        self.decided = False

    async def __call__(self, message: Message) -> None:
        if self.decided:
            await self.send(message)
            return
        if message["type"] == "http.response.start":
            self.start = message
            return
        self.decided = True
        start: Final = self.start
        if start is None:
            await self.send(message)
            return
        start.setdefault("headers", ())
        headers: Final = MutableHeaders(scope=start)
        if (
            message["type"] != "http.response.body"
            or message.get("more_body", False)
            or headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json"
            or "content-encoding" in headers
            or "etag" in headers
            or start["status"] == 206
            or "no-transform" in headers.get("cache-control", "").lower()
        ):
            await self.send(start)
            await self.send(message)
            return
        try:
            body_bytes: Final[bytes] = message.get("body", b"")
            parsed: Final[object] = json.loads(body_bytes)
            body: Final = json.dumps(_display_values(parsed, self.currency.usd_rate), allow_nan=False).encode()
        except (ValidationError, ValueError, OverflowError, RecursionError):
            await self.send(start)
            await self.send(message)
            return
        headers["content-length"] = str(len(body))
        headers["x-litellm-display-currency"] = self.currency.code
        headers["x-litellm-display-currency-rate"] = str(self.currency.usd_rate)
        await self.send(start)
        await self.send({**message, "body": body})

    async def release(self) -> None:
        if not self.decided and self.start is not None:
            await self.send(self.start)


class CurrencyResponseDecorator:
    def __init__(self, app: ASGIApp, get_currency: Callable[[], Currency]) -> None:
        self.app = app
        self.get_currency = get_currency

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] != "GET" or get_route_path(scope) not in MONEY_ROUTES:
            await self.app(scope, receive, send)
            return
        currency: Final = self.get_currency()
        if currency.code == "USD":
            await self.app(scope, receive, send)
            return
        responder: Final = _CurrencyResponder(send, currency)
        await self.app(scope, receive, responder)
        await responder.release()
