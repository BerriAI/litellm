"""Route classification frozen from 8bd7f5089a7500d9c915a5670c1b314ee051f9c9 for independent budget references."""

from __future__ import annotations

import re
from collections.abc import Collection
from typing import Final

from litellm.proxy._types import LiteLLMRoutes

_EU_JEV_DECISION_ROUTE: Final = "/eu_jev/v1/systemone"


class RouteChecks:
    @staticmethod
    def is_llm_api_route(route: str) -> bool:

        if not isinstance(route, str):  # pyright: ignore[reportUnnecessaryIsInstance]  # frozen Phase 1 accepts unvalidated legacy values
            return False

        if route in LiteLLMRoutes.openai_routes.value:
            return True

        if route in LiteLLMRoutes.anthropic_routes.value:
            return True

        if route in LiteLLMRoutes.google_routes.value:
            return True

        if RouteChecks.check_route_access(route=route, allowed_routes=LiteLLMRoutes.mcp_inference_routes.value):
            return True

        if RouteChecks.check_route_access(route=route, allowed_routes=LiteLLMRoutes.agent_inference_routes.value):
            return True

        if RouteChecks.check_route_access(route=route, allowed_routes=LiteLLMRoutes.litellm_native_routes.value):
            return True

        for openai_route in LiteLLMRoutes.openai_routes.value:
            if "{" in openai_route:
                if RouteChecks._route_matches_pattern(route=route, pattern=openai_route):
                    return True

            if RouteChecks.is_wildcard_pattern(pattern=openai_route):
                if RouteChecks.route_matches_wildcard_pattern(route=route, pattern=openai_route):
                    return True

        for google_route in LiteLLMRoutes.google_routes.value:
            if "{" in google_route:
                if RouteChecks._route_matches_pattern(route=route, pattern=google_route):
                    return True

        for anthropic_route in LiteLLMRoutes.anthropic_routes.value:
            if "{" in anthropic_route:
                if RouteChecks._route_matches_pattern(route=route, pattern=anthropic_route):
                    return True

        if RouteChecks._is_azure_openai_route(route=route):
            return True

        for _llm_passthrough_route in LiteLLMRoutes.mapped_pass_through_routes.value:
            if route == _llm_passthrough_route or route.startswith(_llm_passthrough_route + "/"):
                return True
        return RouteChecks.is_builtin_eu_jev_decision_route(route=route)

    @staticmethod
    def is_builtin_eu_jev_decision_route(route: str) -> bool:
        if route.rstrip("/") != _EU_JEV_DECISION_ROUTE:
            return False
        from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (  # noqa: PLC0415  # pass_through_endpoints imports the auth stack
            InitPassThroughEndpointHelpers,
        )

        return not InitPassThroughEndpointHelpers.has_configured_route_ignoring_trailing_slash(
            route=_EU_JEV_DECISION_ROUTE
        )

    @staticmethod
    def is_info_route(route: str) -> bool:

        return route in LiteLLMRoutes.info_routes.value

    @staticmethod
    def _is_azure_openai_route(route: str) -> bool:

        if not isinstance(route, str):  # pyright: ignore[reportUnnecessaryIsInstance]  # frozen Phase 1 accepts unvalidated legacy values
            return False

        deployment_pattern: Final = r"^/openai/deployments/[^/]+/[^/]+/chat/completions$"
        engine_pattern: Final = r"^/engines/[^/]+/chat/completions$"

        if re.match(deployment_pattern, route) or re.match(engine_pattern, route):
            return True
        return False

    @staticmethod
    def _route_matches_pattern(route: str, pattern: str) -> bool:

        if not isinstance(route, str):  # pyright: ignore[reportUnnecessaryIsInstance]  # frozen Phase 1 accepts unvalidated legacy values
            return False

        def _placeholder_to_regex(match: re.Match[str]) -> str:
            placeholder: Final = match.group(0).strip("{}")
            if not placeholder.endswith(":path"):
                return r"[^/]+"

            return r"[^:]+" if ":" in match.string[match.end() :] else r"[\s\S]+"

        pattern = re.sub(r"\{[^}]+\}", _placeholder_to_regex, pattern)  # rebind-ok: preserve Phase 1 mutation order and identity

        pattern = f"^{pattern}$"  # rebind-ok: preserve Phase 1 mutation order and identity
        if re.match(pattern, route):
            return True
        return False

    @staticmethod
    def is_wildcard_pattern(pattern: str) -> bool:

        return pattern.endswith("*")

    @staticmethod
    def route_matches_wildcard_pattern(route: str, pattern: str) -> bool:

        if pattern.endswith("*"):
            prefix: Final = pattern[:-1]
            return route.startswith(prefix)
        else:
            return route == pattern

    @staticmethod
    def check_route_access(route: str, allowed_routes: Collection[str]) -> bool:

        if route in allowed_routes:
            return True

        if any(
            RouteChecks.route_matches_wildcard_pattern(route=route, pattern=allowed_route)
            for allowed_route in allowed_routes
            if RouteChecks.is_wildcard_pattern(pattern=allowed_route)
        ):
            return True

        if any(
            RouteChecks._route_matches_pattern(route=route, pattern=allowed_route) for allowed_route in allowed_routes
        ):
            return True

        return False


__all__ = ()
