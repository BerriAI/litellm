from typing import Final
from unittest.mock import patch

import httpx
import pytest
import respx

from litellm.llms.azure.model_listing import _discovery_cache, discover_azure_openai_deployments

_RESOURCE_ID: Final = (
    "/subscriptions/00000000-0000-0000-0000-000000000002/resourceGroups/test-group"
    "/providers/Microsoft.CognitiveServices/accounts/discovery-test"
)
_API_BASE: Final = "https://discovery-test.openai.azure.com/openai/v1"
_MANAGEMENT: Final = f"https://management.azure.com{_RESOURCE_ID}/deployments"


@pytest.fixture(autouse=True)
def management_resource(monkeypatch: pytest.MonkeyPatch) -> None:
    _discovery_cache.cache_clear()
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_DISCOVERY_RESOURCE_ID", _RESOURCE_ID)
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_DISCOVERY_TTL_SECONDS", "10")


def test_deployment_discovery_pages_caches_and_recovers_after_refresh_failure() -> None:
    page_one: Final = f"{_MANAGEMENT}?api-version=2024-10-01"
    page_two: Final = f"{page_one}&skipToken=next"
    initial: Final = ("first", "second")
    with (
        patch(
            "litellm.llms.azure.model_listing.cached_azure_ad_token_refresh_provider",
            return_value=lambda: "management-token",
        ),
        patch("litellm.llms.azure.model_listing.monotonic") as clock,
        respx.mock as azure,
    ):
        first_request: Final = azure.get(page_one).mock(
            side_effect=[
                httpx.Response(200, json={"value": [{"name": "first"}], "nextLink": page_two}),
                httpx.Response(503),
                httpx.Response(200, json={"value": []}),
            ]
        )
        second_request: Final = azure.get(page_two).respond(
            200, json={"value": [{"name": "first"}, {"name": "second"}]}
        )

        clock.return_value = 100.0
        assert discover_azure_openai_deployments(_API_BASE) == initial
        assert discover_azure_openai_deployments(_API_BASE) == initial
        assert (first_request.call_count, second_request.call_count) == (1, 1)

        clock.return_value = 110.0
        assert discover_azure_openai_deployments(_API_BASE) == initial
        assert discover_azure_openai_deployments(_API_BASE) == initial
        assert first_request.call_count == 2

        clock.return_value = 120.0
        assert discover_azure_openai_deployments(_API_BASE) == ()
        assert first_request.call_count == 3


def test_deployment_discovery_never_follows_a_continuation_to_another_origin() -> None:
    unexpected: Final = "https://other.example/deployments"
    with (
        patch(
            "litellm.llms.azure.model_listing.cached_azure_ad_token_refresh_provider",
            return_value=lambda: "management-token",
        ),
        respx.mock as azure,
    ):
        azure.get(_MANAGEMENT).respond(200, json={"value": [{"name": "first"}], "nextLink": unexpected})
        assert discover_azure_openai_deployments(_API_BASE) is None
        assert len(azure.calls) == 1
