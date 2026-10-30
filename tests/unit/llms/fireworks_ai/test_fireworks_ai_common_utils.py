from typing import Final

import pytest

from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap
from litellm.llms.fireworks_ai.common_utils import ROUTER_SHORT_NAMES, resolve_fireworks_resource_name


@pytest.mark.parametrize(
    "model, expected",
    [
        ("routers/glm-latest", "accounts/fireworks/routers/glm-latest"),
        ("routers/firerouter", "accounts/fireworks/routers/firerouter"),
        ("fireworks_ai/routers/glm-latest", "accounts/fireworks/routers/glm-latest"),
        ("models/glm-4p6", "accounts/fireworks/models/glm-4p6"),
        ("fireworks_ai/models/glm-4p6", "accounts/fireworks/models/glm-4p6"),
        ("glm-4p6", "accounts/fireworks/models/glm-4p6"),
        ("fireworks_ai/glm-4p6", "accounts/fireworks/models/glm-4p6"),
        ("kimi-k2p6-fast", "accounts/fireworks/routers/kimi-k2p6-fast"),
        ("firerouter", "accounts/fireworks/routers/firerouter"),
        ("fireworks_ai/firerouter", "accounts/fireworks/routers/firerouter"),
        ("firerouter/kimi-k3/deepseek-v4", "accounts/fireworks/routers/firerouter/kimi-k3/deepseek-v4"),
        ("firerouter-v2", "accounts/fireworks/models/firerouter-v2"),
        ("auto", "accounts/fireworks/routers/auto"),
        ("fireworks_ai/auto", "accounts/fireworks/routers/auto"),
        ("auto-instant", "accounts/fireworks/routers/auto-instant"),
        ("fireworks_ai/auto-instant", "accounts/fireworks/routers/auto-instant"),
        ("firerouter/auto", "accounts/fireworks/routers/firerouter/auto"),
        ("autoglm-9b", "accounts/fireworks/models/autoglm-9b"),
        ("auto-v2", "accounts/fireworks/models/auto-v2"),
        (
            "accounts/fireworks/routers/glm-latest",
            "accounts/fireworks/routers/glm-latest",
        ),
        (
            "accounts/fireworks/models/glm-4p6",
            "accounts/fireworks/models/glm-4p6",
        ),
        (
            "fireworks_ai/accounts/fireworks/routers/glm-latest",
            "accounts/fireworks/routers/glm-latest",
        ),
        (
            "accounts/fireworks/models/qwen2p5-coder-7b#accounts/gitlab/deployments/2fb7764c",
            "accounts/fireworks/models/qwen2p5-coder-7b#accounts/gitlab/deployments/2fb7764c",
        ),
        (
            "glm-4p6#accounts/gitlab/deployments/2fb7764c",
            "glm-4p6#accounts/gitlab/deployments/2fb7764c",
        ),
        ("FW-Kimi-K3", "FW-Kimi-K3"),
        ("fireworks_ai/FW-Kimi-K3", "FW-Kimi-K3"),
        ("FW-GLM-5.2-Fast", "FW-GLM-5.2-Fast"),
    ],
)
def test_resolve_fireworks_resource_name(model, expected):
    assert resolve_fireworks_resource_name(model) == expected


def _priced_fireworks_names(kind: str) -> frozenset[str]:
    prefix: Final = f"fireworks_ai/accounts/fireworks/{kind}/"
    catalog: Final = GetModelCostMap.loaded_model_cost_map()
    return frozenset(key.removeprefix(prefix) for key in catalog if key.startswith(prefix))


def test_a_short_name_priced_only_as_a_router_resolves_to_the_router():
    router_only: Final = _priced_fireworks_names("routers") - _priced_fireworks_names("models")
    assert router_only
    assert {name: resolve_fireworks_resource_name(f"fireworks_ai/{name}") for name in router_only} == {
        name: f"accounts/fireworks/routers/{name}" for name in router_only
    }


def test_a_short_name_priced_as_a_model_keeps_the_model_path():
    model_names: Final = frozenset(
        name
        for name in _priced_fireworks_names("models")
        if not name.endswith("-fast") and name not in ROUTER_SHORT_NAMES
    )
    assert model_names
    assert {name: resolve_fireworks_resource_name(f"fireworks_ai/{name}") for name in model_names} == {
        name: f"accounts/fireworks/models/{name}" for name in model_names
    }
