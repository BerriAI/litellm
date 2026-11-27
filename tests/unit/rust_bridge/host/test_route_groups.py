from pathlib import Path
from typing import Final

from pydantic import TypeAdapter

from litellm.proxy._types import LiteLLMRoutes
from litellm.rust_bridge.host.route_groups import route_groups

CHECKED_IN: Final = Path(__file__).parents[4] / "litellm/rust/crates/gateway-auth-core/src/routes/litellm_routes.json"
REGENERATE: Final = "run litellm/rust/crates/gateway-auth-core/scripts/generate_litellm_routes.py"
AS_IMPORTED: Final = {
    member.name: sorted(member.value) if isinstance(member.value, frozenset) else list(member.value)
    for member in LiteLLMRoutes
}


def test_route_groups_hand_rust_every_litellm_route_group_by_name():
    groups = route_groups()

    assert {name: set(routes) for name, routes in groups.items()} == {
        member.name: set(member.value) for member in LiteLLMRoutes
    }


def test_the_route_groups_checked_in_for_the_gateway_are_the_litellm_route_groups():
    checked_in: Final = TypeAdapter(dict[str, list[str]]).validate_json(CHECKED_IN.read_text())

    assert checked_in == AS_IMPORTED, REGENERATE
