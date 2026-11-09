from collections.abc import Callable
from typing import Final, Literal

import pytest

from litellm._auth.route_enable import (
    LlmRouteDecision,
    RouteEnableDecision,
    RouteEnableFacts,
    RouteSetting,
    llm_route,
    route_enable,
)
from litellm.proxy.auth.route_policy_adapter import route_context
from tests.unit._auth.support.route_enable_reference import route_enable as reference


@pytest.mark.parametrize("policy", (reference, route_enable))
@pytest.mark.parametrize(
    "route", ("/unknown", "/models", "/v1/models", "/chat/completions", "/key/generate", "/user/info")
)
@pytest.mark.parametrize("imported", (False, True))
@pytest.mark.parametrize("failed", (None, "Other", "Http"))
@pytest.mark.parametrize("exemptions", (None, (), ("/models", "/v1/models")))
@pytest.mark.parametrize(
    "setting",
    (
        None,
        RouteSetting(True, False),
        RouteSetting(False, False),
        RouteSetting(True, True),
        RouteSetting(True, True, True),
        RouteSetting(True, True, False),
    ),
)
def test_optional_route_enable_decision_table(
    policy: Callable[[RouteEnableFacts], RouteEnableDecision],
    route: str,
    imported: bool,
    failed: Literal["Other", "Http"] | None,
    exemptions: tuple[str, ...] | None,
    setting: RouteSetting | None,
) -> None:
    facts: Final = RouteEnableFacts(route_context(route), imported, failed, setting, setting, exemptions)
    assert policy(facts) == reference(facts)


@pytest.mark.parametrize(
    ("route", "configured", "expected"),
    (
        ("/chat/completions", None, "Allow"),
        ("/unknown", None, "Deny"),
        ("/eu_jev/v1/systemone", None, "ReadEuJev"),
        ("/eu_jev/v1/systemone", False, "Allow"),
        ("/eu_jev/v1/systemone", True, "Deny"),
        ("/eu_jev/v1/systemone/", True, "Deny"),
    ),
)
def test_llm_route_configuration_read_is_selected_only_at_builtin_fallback(
    route: str, configured: bool | None, expected: LlmRouteDecision
) -> None:
    assert llm_route(route_context(route), configured) == expected


@pytest.mark.parametrize("configured", (None, False, True))
@pytest.mark.parametrize("route", ("/eu_jev/v1/systemone", "/eu_jev/v1/systemone/"))
def test_optional_route_builtin_configuration_matches_reference(configured: bool | None, route: str) -> None:
    facts: Final = RouteEnableFacts(
        route_context(route),
        imported=True,
        llm=RouteSetting(True, True, True),
        exemptions=(),
        eu_jev_configured=configured,
    )
    assert route_enable(facts) == reference(facts)
