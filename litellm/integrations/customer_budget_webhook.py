import os
from collections.abc import Mapping
from typing import Final

from pydantic import BaseModel, ConfigDict, HttpUrl, TypeAdapter

from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # Legacy params are untyped.
    httpxSpecialProvider,
)
from litellm.types.integrations.customer_budget import CustomerBudgetEvent


class _Settings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    endpoint: HttpUrl | None = None
    headers: Mapping[str, str] | None = None


_URL: Final = TypeAdapter(HttpUrl)
_HEADERS: Final = TypeAdapter(Mapping[str, str])


class CustomerBudgetWebhook(CustomLogger):
    def __init__(
        self,
        endpoint: str | None = None,
        headers: Mapping[str, str] | None = None,
        *,
        async_httpx_client: AsyncHTTPHandler | None = None,
    ) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  # Base logger kwargs are untyped.
        self.endpoint: Final = str(_URL.validate_python(endpoint or os.environ.get("CUSTOMER_BUDGET_WEBHOOK_URL")))
        self.headers: Final = {
            "Content-Type": "application/json",
            **_HEADERS.validate_json(os.environ.get("CUSTOMER_BUDGET_WEBHOOK_HEADERS", "{}")),
            **(headers or {}),
        }
        self.async_httpx_client: Final = (
            async_httpx_client
            if async_httpx_client is not None
            else get_async_httpx_client(llm_provider=httpxSpecialProvider.LoggingCallback)
        )

    @staticmethod
    def from_config(value: object) -> "CustomerBudgetWebhook":
        settings: Final = _Settings.model_validate(value or {})
        return CustomerBudgetWebhook(
            endpoint=str(settings.endpoint) if settings.endpoint is not None else None,
            headers=settings.headers,
        )

    async def async_on_customer_budget_event(self, event: CustomerBudgetEvent) -> None:
        client: Final = self.async_httpx_client
        response: Final = await client.post(  # pyright: ignore[reportUnknownMemberType]  # Legacy HTTP params are untyped.
            url=self.endpoint, headers=dict(self.headers), json=event.model_dump(mode="json"), timeout=10
        )
        response.raise_for_status()
