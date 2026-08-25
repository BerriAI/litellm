"""Live e2e: a deployment's custom pricing override can be cleared, and Reload
Price Data restores cost-map rates (ticket #6844).

Models created through the UI carry the displayed price as a deployment-level
override, which then shadows the shared cost map forever: Reload Price Data
becomes a no-op for that deployment and the admin cannot get back to canonical
pricing. The escape hatch is PATCH /model/{id}/update with the pricing field set
to an explicit null, which drops the override from both stored blobs
(SPECIAL_MODEL_INFO_PARAMS clears in update_db_model). These tests pin that
escape hatch and the reload endpoint the UI's Reload Price Data button calls.

Known gap called out in the PR: the older POST /model/update route ignores
explicit nulls (its merge keeps the stored value for any null field), so only
the PATCH route can clear an override.
"""

from __future__ import annotations

import time

import pytest
from pydantic import BaseModel

from e2e_config import unique_marker
from e2e_http import NoBody, unwrap
from lifecycle import ResourceManager
from models import LiteLLMParamsBody, ModelInfoEntry
from proxy_client import ProxyClient

pytestmark = pytest.mark.e2e

BACKEND_MODEL = "gemini/gemini-2.5-flash"
# Deliberately ~100x the canonical gemini-2.5-flash input rate (3e-7) so an
# override that fails to clear is unmistakable from the cost-map value.
OVERRIDE_INPUT_RATE = 5e-05


class PricingPatchBody(BaseModel):
    """PATCH /model/{id}/update body carrying an explicit-null pricing clear.

    The pricing fields must serialize as literal nulls (the handler only clears
    fields present in the payload), so they cannot ride LiteLLMParamsBody, whose
    None fields are dropped by the transport's exclude_none serialization.
    """

    litellm_params: dict[str, float | None]


def _register_overridden_model(proxy: ProxyClient, resources: ResourceManager) -> tuple[str, str]:
    model_name = f"e2e-pricing-override-{unique_marker()}"
    model_id = proxy.create_model(
        model_name,
        LiteLLMParamsBody(
            model=BACKEND_MODEL,
            api_key="os.environ/GEMINI_API_KEY",
            input_cost_per_token=OVERRIDE_INPUT_RATE,
        ),
    )
    resources.defer(lambda: proxy.delete_model(model_id))
    return model_name, model_id


def _model_entry(proxy: ProxyClient, model_name: str) -> ModelInfoEntry:
    for entry in proxy.model_info():
        if entry.model_name == model_name:
            return entry
    pytest.fail(f"{model_name} absent from /model/info")


def _poll_override_cleared(proxy: ProxyClient, model_name: str) -> ModelInfoEntry:
    """Poll /model/info until the deployment's input override reads as cleared
    (the router picks the change up on its periodic DB reload)."""
    deadline = time.monotonic() + proxy.poll_timeout
    while True:
        entry = _model_entry(proxy, model_name)
        if entry.litellm_params.input_cost_per_token is None:
            return entry
        if time.monotonic() >= deadline:
            return entry
        time.sleep(proxy.poll_interval)


class TestPricingOverrideClearAndReload:
    @pytest.mark.covers("mgmt.model.update.clears_pricing_override")
    def test_explicit_null_patch_clears_pricing_override(
        self, proxy: ProxyClient, resources: ResourceManager
    ) -> None:
        model_name, model_id = _register_overridden_model(proxy, resources)
        assert _model_entry(proxy, model_name).litellm_params.input_cost_per_token == OVERRIDE_INPUT_RATE

        unwrap(
            proxy.transport.patch(
                f"/model/{model_id}/update",
                headers=proxy.transport.master,
                json=PricingPatchBody(litellm_params={"input_cost_per_token": None}),
                response_type=NoBody,
            )
        )

        entry = _poll_override_cleared(proxy, model_name)
        assert entry.litellm_params.input_cost_per_token is None, (
            f"PATCH with an explicit null must drop the stored pricing override; the "
            f"deployment still bills at {entry.litellm_params.input_cost_per_token} "
            f"(ticket #6844: stale UI-pinned prices cannot be cleared)"
        )
        resolved_rate = entry.model_info.input_cost_per_token
        assert resolved_rate is not None and resolved_rate < OVERRIDE_INPUT_RATE, (
            f"after the clear the deployment must fall back to the cost-map rate, "
            f"not keep the {OVERRIDE_INPUT_RATE} override: resolved {resolved_rate}"
        )

    @pytest.mark.covers("mgmt.model.cost_map_reload.keeps_cleared_override")
    def test_reload_price_data_does_not_overwrite_cleared_deployment(
        self, proxy: ProxyClient, resources: ResourceManager
    ) -> None:
        model_name, model_id = _register_overridden_model(proxy, resources)
        unwrap(
            proxy.transport.patch(
                f"/model/{model_id}/update",
                headers=proxy.transport.master,
                json=PricingPatchBody(litellm_params={"input_cost_per_token": None}),
                response_type=NoBody,
            )
        )
        cleared_entry = _poll_override_cleared(proxy, model_name)
        assert cleared_entry.litellm_params.input_cost_per_token is None, (
            f"override must be cleared before the reload, else a later ordinary router refresh "
            f"could clear it and let this test pass without proving reload preserved the clear: "
            f"{cleared_entry.litellm_params.input_cost_per_token}"
        )

        unwrap(
            proxy.transport.post(
                "/reload/model_cost_map",
                headers=proxy.transport.master,
                json=NoBody(),
                response_type=NoBody,
            )
        )

        entry = _poll_override_cleared(proxy, model_name)
        assert entry.litellm_params.input_cost_per_token is None, (
            f"Reload Price Data resurrected the cleared deployment override: "
            f"{entry.litellm_params.input_cost_per_token}"
        )
        resolved_rate = entry.model_info.input_cost_per_token
        assert resolved_rate is not None and resolved_rate < OVERRIDE_INPUT_RATE, (
            f"after the reload the deployment must price at the cost-map rate: {resolved_rate}"
        )
