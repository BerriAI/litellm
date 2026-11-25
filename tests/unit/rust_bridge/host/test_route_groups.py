from litellm.proxy._types import LiteLLMRoutes
from litellm.rust_bridge.host.route_groups import route_groups


def test_route_groups_hand_rust_every_litellm_route_group_by_name():
    groups = route_groups()

    assert {name: set(routes) for name, routes in groups.items()} == {
        member.name: set(member.value) for member in LiteLLMRoutes
    }
