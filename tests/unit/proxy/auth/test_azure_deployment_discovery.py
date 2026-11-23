from typing import Final
from unittest.mock import patch

import pytest
import respx

import litellm
from litellm import Router
from litellm.llms.azure.model_listing import _discovery_cache
from litellm.proxy.auth.model_checks import get_complete_model_list
from litellm.utils import AZURE_MODEL_PLACEHOLDER


@pytest.mark.parametrize(
    ("enabled", "matching_resource", "expected"),
    [
        (True, True, ["azure/blue-chat", "azure/embedding-east"]),
        (False, True, [f"azure/{AZURE_MODEL_PLACEHOLDER}"]),
        (True, False, [f"azure/{AZURE_MODEL_PLACEHOLDER}"]),
    ],
)
def test_azure_wildcard_lists_only_discoverable_deployments(
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
    matching_resource: bool,
    expected: list[str],
) -> None:
    resource_id: Final = (
        "/subscriptions/00000000-0000-0000-0000-000000000001/resourceGroups/group"
        "/providers/Microsoft.CognitiveServices/accounts/proxy-test"
    )
    account: Final = "proxy-test" if matching_resource else "other-test"
    router: Final = Router(
        model_list=[
            {
                "model_name": "azure/*",
                "litellm_params": {
                    "model": "azure/*",
                    "api_base": f"https://{account}.openai.azure.com",
                    "api_key": "inference-key",
                },
            }
        ]
    )
    _discovery_cache.cache_clear()
    monkeypatch.setattr(litellm, "check_provider_endpoint", enabled)
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_DISCOVERY_RESOURCE_ID", resource_id)
    with (
        patch(
            "litellm.llms.azure.model_listing.cached_azure_ad_token_refresh_provider",
            return_value=lambda: "management-token",
        ),
        respx.mock as azure,
    ):
        management: Final = azure.get(f"https://management.azure.com{resource_id}/deployments").respond(
            200, json={"value": [{"name": "blue-chat"}, {"name": "embedding-east"}]}
        )
        listed: Final = get_complete_model_list(
            key_models=["azure/*"],
            team_models=[],
            proxy_model_list=["azure/*"],
            user_model=None,
            infer_model_from_keys=False,
            llm_router=router,
        )

        assert listed == expected
        assert management.call_count == int(enabled and matching_resource)
        if management.called:
            assert management.calls[0].request.headers["Authorization"] == "Bearer management-token"
