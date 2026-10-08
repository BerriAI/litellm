from litellm.proxy._types import LiteLLMRoutes
from litellm.proxy.agent_endpoints.auth.managed_authorization import _MANAGED_MODEL_ROUTES
from litellm.proxy.auth.auth_utils import normalize_request_route
from litellm.proxy.common_utils.custom_openapi_spec import CustomOpenAPISpec
from litellm.proxy.response_api_endpoints.endpoints import router
from litellm.proxy.spend_tracking.budget_reservation import _UNBILLED_ROUTES
from litellm.types.utils import API_ROUTE_TO_CALL_TYPES, CallTypes


def test_openai_responses_routes_registered():
    """Verify that Azure-style /openai/responses endpoints are registered on the router."""
    routes = {getattr(r, "path", None) for r in router.routes}

    expected_routes = [
        "/openai/responses",
        "/openai/responses/{response_id}",
        "/openai/responses/{response_id}/input_items",
        "/openai/responses/compact",
        "/openai/responses/input_tokens",
        "/openai/responses/{response_id}/cancel",
    ]

    for expected in expected_routes:
        assert expected in routes, f"Route {expected} missing from router.routes!"

    assert "/openai/responses" in CustomOpenAPISpec.RESPONSES_API_PATHS
    assert "/openai/responses" in _MANAGED_MODEL_ROUTES
    assert "/openai/responses/input_tokens" in _UNBILLED_ROUTES
    assert "/openai/responses" in LiteLLMRoutes.openai_routes.value


def test_openai_responses_api_route_to_call_types():
    """Verify that /openai/responses mappings exist in API_ROUTE_TO_CALL_TYPES for guardrails/tools."""
    assert API_ROUTE_TO_CALL_TYPES.get("/openai/responses") == (CallTypes.aresponses, CallTypes.responses)
    assert API_ROUTE_TO_CALL_TYPES.get("/openai/responses/{response_id}") == (CallTypes.aresponses, CallTypes.responses)
    assert API_ROUTE_TO_CALL_TYPES.get("/openai/responses/{response_id}/input_items") == (CallTypes.alist_input_items,)


def test_openai_responses_metrics_normalization():
    """Verify that dynamic response IDs are normalized to prevent Prometheus high-cardinality metrics."""
    assert normalize_request_route("/openai/responses/resp-123456") == "/openai/responses/{response_id}"
    assert normalize_request_route("/openai/responses/resp-123456/cancel") == "/openai/responses/{response_id}/cancel"
    assert normalize_request_route("/openai/responses/resp-123456/input_items") == "/openai/responses/{response_id}/input_items"
