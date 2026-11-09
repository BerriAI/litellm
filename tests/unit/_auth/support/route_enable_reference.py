from typing import Final

from litellm._auth.route_enable import RouteEnableDecision, RouteEnableFacts, RouteFeature, RouteSetting
from tests.unit._auth.support.route_reference import RouteChecks


def _setting(facts: RouteSetting | None, feature: RouteFeature) -> RouteEnableDecision | None:
    if facts is None:
        return RouteEnableDecision("ReadSettings", feature)
    if facts.environment_present and not facts.premium:
        return RouteEnableDecision("Deny", feature, "License")
    if facts.secret is None:
        return RouteEnableDecision("ReadSecret", feature)
    if facts.secret:
        return RouteEnableDecision("Deny", feature, "Disabled")
    return None


def route_enable(facts: RouteEnableFacts) -> RouteEnableDecision:
    if facts.failed is not None:
        return RouteEnableDecision("PropagateHttp" if facts.failed == "Http" else "Allow")
    if not facts.imported:
        return RouteEnableDecision("ReadImport")
    if RouteChecks.check_route_access(facts.context.route, facts.context.catalog.groups.get("management_routes", ())):
        admin: Final = _setting(facts.admin, "Admin")
        if admin is not None:
            return admin
    if not RouteChecks.is_llm_api_route(facts.context.route, configured_eu_jev=True):
        if facts.context.route.rstrip("/") != "/eu_jev/v1/systemone":
            return RouteEnableDecision("Allow")
        if facts.eu_jev_configured is None:
            return RouteEnableDecision("ReadEuJev")
        if facts.eu_jev_configured:
            return RouteEnableDecision("Allow")
    if facts.exemptions is None:
        return RouteEnableDecision("ReadExemptions")
    if facts.context.route in facts.exemptions:
        return RouteEnableDecision("Allow")
    return _setting(facts.llm, "Llm") or RouteEnableDecision("Allow")
