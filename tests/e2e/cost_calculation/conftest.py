"""Cost-calculation suite fixtures.

Runs against a dedicated proxy whose whole model cost map is the test-owned
``tests/e2e/cost_map.json`` (LITELLM_MODEL_COST_MAP_URL), so every deployment
bills at rates the test asserts literal arithmetic on. Provider calls are
answered by the scripted-provider sidecar (``scripted_provider.py``), registered
per scenario over its control API.

Deselected unless E2E_COST_MAP_STACK is set (marker `cost_map_stack`).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Final, Protocol, cast

import pytest

from cost_matrix import Case, FrontierModel
from e2e_config import COST_MAP_PROXY_URL, SCRIPTED_PROVIDER_PROXY_BASE
from lifecycle import ResourceManager
from models import LiteLLMParamsBody, ModelInfoBody, ModelNewBody
from proxy_client import ProxyClient, build_proxy_client
from scripted_client import ScenarioHandle, delete_scenario, register_scenario
from scripted_provider import Scenario


def _load_cost_rows() -> ModuleType:
    """Load quota_management/spend_tracking/cost_rows.py by path (the e2e tree
    has no package layout), the same trick the mcp suite uses for
    logging/datadog_reader.py."""
    path: Final = (
        Path(__file__).resolve().parent.parent
        / "quota_management"
        / "spend_tracking"
        / "cost_rows.py"
    )
    name: Final = "e2e_spend_tracking_cost_rows"
    spec: Final = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module: Final = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class SpendCostBreakdown(Protocol):
    input_cost: float | None
    output_cost: float | None
    cache_read_cost: float | None
    cache_creation_cost: float | None
    reasoning_cost: float | None
    tool_usage_cost: float | None
    total_cost: float | None
    service_tier: str | None

    def model_dump(self) -> Mapping[str, object]: ...


class SpendRowMetadata(Protocol):
    cost_breakdown: SpendCostBreakdown | None


class SpendCostRow(Protocol):
    """The slice of spend_tracking.cost_rows.CostRow this suite reads."""

    spend: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    metadata: SpendRowMetadata | None

    @property
    def breakdown(self) -> SpendCostBreakdown: ...


class CostRowsModule(Protocol):
    """cost_rows.py loaded by path has no importable name for basedpyright, so
    its surface is declared here and reached through a single cast."""

    approx_equal: Callable[[float, float], bool]
    assert_total_is_sum_of_components: Callable[[SpendCostRow], None]
    poll_cost_row_where: Callable[
        [ProxyClient, str, Callable[[SpendCostRow], bool]], SpendCostRow | None
    ]


cost_rows: Final[CostRowsModule] = cast(  # cast-ok: cost_rows.py is loaded by path, so basedpyright has no importable name for it; its surface is declared in CostRowsModule
    CostRowsModule, _load_cost_rows()
)


@dataclass(frozen=True, slots=True)
class CostCalcClient:
    """The suite's client: a ProxyClient pointed at the cost-map proxy pod."""

    proxy: ProxyClient


@pytest.fixture(scope="session")
def client() -> CostCalcClient:
    proxy: Final = build_proxy_client(
        base_url=COST_MAP_PROXY_URL,
        control_plane_base_url=COST_MAP_PROXY_URL,
        replica_urls=(COST_MAP_PROXY_URL,),
    )
    return CostCalcClient(proxy=proxy)


_vertex_key_pem: str | None = None


def _vertex_service_account_json() -> str:
    """A service-account credential JSON whose token_uri is the sidecar's
    /_oauth/token route: the proxy's google-auth refresh then gets a scripted
    access token without touching Google. One generated RSA key per process."""
    global _vertex_key_pem  # mutable-ok: session-scoped key generation cached for reuse
    if _vertex_key_pem is None:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import rsa

        _vertex_key_pem = (
            rsa.generate_private_key(public_exponent=65537, key_size=2048)
            .private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
            .decode()
        )
    return json.dumps(
        {
            "type": "service_account",
            "project_id": "cc-scripted-project",
            "private_key_id": "scripted",
            "private_key": _vertex_key_pem,
            "client_email": "scripted@cc-scripted-project.iam.gserviceaccount.com",
            "client_id": "0",
            "auth_uri": f"{SCRIPTED_PROVIDER_PROXY_BASE}/_oauth/authorize",
            "token_uri": f"{SCRIPTED_PROVIDER_PROXY_BASE}/_oauth/token",
        }
    )


def register_scenario_deployment(
    client: CostCalcClient,
    resources: ResourceManager,
    model: FrontierModel,
    case: Case,
    marker: str,
) -> tuple[str, ScenarioHandle]:
    """Register the case's scenario on the sidecar plus a deployment pointed at
    it; both are torn down by ``resources``. Returns the callable model_name."""
    scenario: Final[Scenario] = case.scenario(
        scenario_id=f"sc-{marker}", model=model, text=f"scripted answer {marker}"
    )
    handle: Final = register_scenario(scenario)
    resources.defer(lambda: delete_scenario(handle))
    model_name: Final = f"{model.model_name}-{marker}"
    extra_params: Final[dict[str, str]] = dict(model.litellm_params)
    if model.wire == "vertex_generate":
        extra_params["vertex_credentials"] = _vertex_service_account_json()
    model_id: Final = client.proxy.register_model(
        ModelNewBody(
            model_name=model_name,
            litellm_params=LiteLLMParamsBody.model_validate(
                {
                    "model": model.litellm_model,
                    "api_key": model.api_key,
                    "api_base": handle.api_base(),
                    **extra_params,
                }
            ),
            model_info=ModelInfoBody(base_model=model.base_model),
        )
    )
    resources.defer(lambda: client.proxy.delete_model(model_id))
    return model_name, handle
