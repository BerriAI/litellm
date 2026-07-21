"""Thin async client for Azure Cost Management REST API.

Used by the PTU reservation rollup to fetch billed cost for a specific
Azure resource id on a given day (LIT-4077). Deliberately narrow: exposes
one public method the rollup needs, not a general Cost Management SDK.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Callable, Optional

import httpx

from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, get_async_httpx_client
from litellm.types.llms.custom_http import httpxSpecialProvider


class AzureCostManagementError(Exception):
    """Raised when the Cost Management API returns a non-2xx, network failure, or an unexpected payload."""


@dataclass(frozen=True, slots=True)
class AzureCostManagementConfig:
    subscription_id: str
    tenant_id: str
    client_id: str
    client_secret: str = field(repr=False)
    api_version: str = "2023-11-01"
    management_base_url: str = "https://management.azure.com"

    @classmethod
    def from_env(cls, subscription_id: str) -> AzureCostManagementConfig:
        tenant_id = os.getenv("AZURE_TENANT_ID")
        client_id = os.getenv("AZURE_CLIENT_ID")
        client_secret = os.getenv("AZURE_CLIENT_SECRET")
        missing = [
            name
            for name, value in (
                ("AZURE_TENANT_ID", tenant_id),
                ("AZURE_CLIENT_ID", client_id),
                ("AZURE_CLIENT_SECRET", client_secret),
            )
            if not value
        ]
        if missing:
            raise AzureCostManagementError(f"Missing required env vars for Azure Cost Management auth: {missing}")
        return cls(
            subscription_id=subscription_id,
            tenant_id=tenant_id or "",
            client_id=client_id or "",
            client_secret=client_secret or "",
            management_base_url=os.getenv("AZURE_MANAGEMENT_BASE_URL", "https://management.azure.com"),
        )


TokenProvider = Callable[[], str]


def _default_token_provider_factory(config: AzureCostManagementConfig) -> TokenProvider:
    from litellm.llms.azure.common_utils import get_azure_ad_token_from_entra_id

    return get_azure_ad_token_from_entra_id(
        tenant_id=config.tenant_id,
        client_id=config.client_id,
        client_secret=config.client_secret,
        scope="https://management.azure.com/.default",
    )


class AzureCostManagementClient:
    """Fetch billed cost for a single Azure resource on a specific day.

    Auth: Entra ID service principal via env vars (``AZURE_TENANT_ID``,
    ``AZURE_CLIENT_ID``, ``AZURE_CLIENT_SECRET``). The token provider is
    dependency-injectable for tests.

    Currency: the API returns cost in the subscription's billing currency,
    exposed on the response. Callers that require USD MUST inspect
    ``last_currency`` after each call and handle non-USD.
    """

    def __init__(
        self,
        config: AzureCostManagementConfig,
        *,
        http_handler: Optional[AsyncHTTPHandler] = None,
        token_provider: Optional[TokenProvider] = None,
    ) -> None:
        self._config = config
        self._http = http_handler or get_async_httpx_client(httpxSpecialProvider.AzureCostManagement)
        self._token_provider = token_provider or _default_token_provider_factory(config)
        self._last_currency: Optional[str] = None

    @property
    def last_currency(self) -> Optional[str]:
        return self._last_currency

    async def get_daily_cost(self, resource_id: str, day: date) -> float:
        """Return billed cost for ``resource_id`` on the UTC calendar day ``day``.

        Raises ``AzureCostManagementError`` on non-2xx responses or unexpected
        payloads. Returns 0.0 when Azure reports no rows for the window (a
        valid response, typically due to reporting lag or zero utilization).
        """
        url = (
            f"{self._config.management_base_url}/subscriptions/"
            f"{self._config.subscription_id}"
            "/providers/Microsoft.CostManagement/query"
        )
        start = datetime.combine(day, time.min, tzinfo=timezone.utc)
        end = start + timedelta(days=1) - timedelta(microseconds=1)
        body = {
            "type": "ActualCost",
            "timeframe": "Custom",
            "timePeriod": {
                "from": start.isoformat().replace("+00:00", "Z"),
                "to": end.isoformat().replace("+00:00", "Z"),
            },
            "dataset": {
                "granularity": "Daily",
                "aggregation": {"totalCost": {"name": "Cost", "function": "Sum"}},
                "filter": {
                    "dimensions": {
                        "name": "ResourceId",
                        "operator": "In",
                        "values": [resource_id],
                    }
                },
            },
        }
        token = self._token_provider()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        try:
            response = await self._http.post(
                url=url,
                json=body,
                params={"api-version": self._config.api_version},
                headers=headers,
            )
        except httpx.HTTPStatusError as exc:
            raise AzureCostManagementError(
                f"Azure Cost Management HTTP {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.RequestError as exc:
            raise AzureCostManagementError(f"Azure Cost Management network error: {exc}") from exc
        return self._parse_cost_and_currency(response.json())

    def _parse_cost_and_currency(self, payload: Any) -> float:
        properties = payload.get("properties", {}) if isinstance(payload, dict) else {}
        rows = properties.get("rows") or []
        columns = properties.get("columns") or []

        if not rows:
            self._last_currency = None
            return 0.0

        column_index = {col.get("name"): i for i, col in enumerate(columns) if isinstance(col, dict)}
        cost_idx: Optional[int] = None
        for candidate in ("Cost", "PreTaxCost", "CostUSD"):
            if candidate in column_index:
                cost_idx = column_index[candidate]
                break
        currency_idx = column_index.get("Currency")
        if cost_idx is None:
            raise AzureCostManagementError(f"No Cost column in response; columns={list(column_index)}")

        try:
            total = sum(float(row[cost_idx]) for row in rows)
        except (TypeError, ValueError, IndexError) as exc:
            raise AzureCostManagementError(f"Non-numeric cost value in rows: {rows}") from exc

        if currency_idx is not None and rows:
            try:
                self._last_currency = str(rows[0][currency_idx])
            except (IndexError, TypeError):
                self._last_currency = None
        else:
            self._last_currency = None

        if self._last_currency is not None and self._last_currency.upper() != "USD":
            verbose_logger.warning(
                "Azure Cost Management returned currency=%s for resource %s; value written as-is without conversion",
                self._last_currency,
                "(resource redacted)",
            )
        return total
