import ast
import inspect
from datetime import datetime
from unittest.mock import MagicMock, patch

from litellm.proxy.pass_through_endpoints.llm_provider_handlers.vertex_passthrough_logging_handler import (
    VertexPassthroughLoggingHandler,
)
from litellm.types.utils import Choices, Message, ModelResponse


def _make_model_response() -> ModelResponse:
    return ModelResponse(
        id="test-id",
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(content="hello", role="assistant"),
            )
        ],
        created=1234567890,
        model="claude-opus-4-8",
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )


class TestVertexPassthroughCustomPricing:
    """Vertex passthrough cost tracking must honor per-deployment custom pricing
    (model_info), mirroring the Anthropic passthrough handler — not fall back to
    the built-in litellm.model_cost map only."""

    def test_generate_content_forwards_custom_pricing_and_router_model_id(self):
        logging_obj = MagicMock()
        logging_obj.model_call_details = {}
        logging_obj.litellm_call_id = "call-1"
        logging_obj.get_router_model_id.return_value = "dep-123"
        # Per-deployment custom pricing set via model_info on the router deployment.
        logging_obj.litellm_params = {
            "metadata": {
                "model_info": {
                    "id": "dep-123",
                    "input_cost_per_token": 1e-5,
                    "output_cost_per_token": 3e-5,
                }
            }
        }

        with patch("litellm.completion_cost", return_value=0.123) as mock_cost:
            VertexPassthroughLoggingHandler._create_vertex_response_logging_payload_for_generate_content(
                litellm_model_response=_make_model_response(),
                model="vertex_ai/claude-opus-4-8",
                kwargs={},
                start_time=datetime.now(),
                end_time=datetime.now(),
                logging_obj=logging_obj,
                custom_llm_provider="vertex_ai",
            )

        assert mock_cost.call_count == 1
        call_kwargs = mock_cost.call_args.kwargs
        assert call_kwargs["custom_pricing"] is True
        assert call_kwargs["router_model_id"] == "dep-123"

    def test_generate_content_without_custom_pricing(self):
        logging_obj = MagicMock()
        logging_obj.model_call_details = {}
        logging_obj.litellm_call_id = "call-2"
        logging_obj.get_router_model_id.return_value = None
        logging_obj.litellm_params = {}

        with patch("litellm.completion_cost", return_value=0.0) as mock_cost:
            VertexPassthroughLoggingHandler._create_vertex_response_logging_payload_for_generate_content(
                litellm_model_response=_make_model_response(),
                model="vertex_ai/claude-opus-4-8",
                kwargs={},
                start_time=datetime.now(),
                end_time=datetime.now(),
                logging_obj=logging_obj,
                custom_llm_provider="vertex_ai",
            )

        assert mock_cost.call_count == 1
        call_kwargs = mock_cost.call_args.kwargs
        assert call_kwargs["custom_pricing"] is False
        assert call_kwargs["router_model_id"] is None


class TestCustomPricingKwargsHelper:
    def test_helper_handles_logging_obj_without_litellm_params(self):
        """getattr fallback: a logging object missing litellm_params must not raise."""
        logging_obj = MagicMock(spec=["get_router_model_id"])
        logging_obj.get_router_model_id.return_value = None

        result = VertexPassthroughLoggingHandler._custom_pricing_kwargs(logging_obj)

        assert result == {"custom_pricing": False, "router_model_id": None}


class TestResolveRouterModelId:
    """Vertex passthrough requests are matched by project/location, not a router
    deployment, so ``get_router_model_id()`` is usually empty. The id must then be
    resolved from the router by model name so per-deployment model_info pricing is
    honored instead of the built-in list price."""

    def test_prefers_id_already_on_logging_obj(self):
        logging_obj = MagicMock()
        logging_obj.get_router_model_id.return_value = "dep-existing"

        result = VertexPassthroughLoggingHandler._resolve_router_model_id(logging_obj, "vertex_ai/gemini-3.5-flash")

        assert result == "dep-existing"

    def test_falls_back_to_router_lookup_by_model_name(self):
        logging_obj = MagicMock()
        logging_obj.get_router_model_id.return_value = None
        mock_router = MagicMock()
        mock_router.get_model_ids.return_value = ["dep-gemini-35"]

        with patch("litellm.proxy.proxy_server.llm_router", mock_router):
            result = VertexPassthroughLoggingHandler._resolve_router_model_id(logging_obj, "vertex_ai/gemini-3.5-flash")

        assert result == "dep-gemini-35"
        mock_router.get_model_ids.assert_called_once()

    def test_returns_none_when_model_not_in_router(self):
        logging_obj = MagicMock()
        logging_obj.get_router_model_id.return_value = None
        mock_router = MagicMock()
        mock_router.get_model_ids.return_value = []

        with patch("litellm.proxy.proxy_server.llm_router", mock_router):
            result = VertexPassthroughLoggingHandler._resolve_router_model_id(logging_obj, "vertex_ai/gemini-3.5-flash")

        assert result is None

    def test_returns_none_when_router_unavailable(self):
        logging_obj = MagicMock()
        logging_obj.get_router_model_id.return_value = None

        with patch("litellm.proxy.proxy_server.llm_router", None):
            result = VertexPassthroughLoggingHandler._resolve_router_model_id(logging_obj, "vertex_ai/gemini-3.5-flash")

        assert result is None

    def test_generate_content_uses_router_fallback_id(self):
        """End-to-end for the Gemini generateContent path: no id on the logging
        object, but the router knows the model -> its deployment id is forwarded
        to completion_cost so model_info pricing applies."""
        logging_obj = MagicMock()
        logging_obj.model_call_details = {}
        logging_obj.litellm_call_id = "call-4"
        logging_obj.get_router_model_id.return_value = None
        logging_obj.litellm_params = {}
        mock_router = MagicMock()
        mock_router.get_model_ids.return_value = ["dep-gemini-35"]

        with (
            patch("litellm.proxy.proxy_server.llm_router", mock_router),
            patch("litellm.completion_cost", return_value=0.001) as mock_cost,
        ):
            VertexPassthroughLoggingHandler._create_vertex_response_logging_payload_for_generate_content(
                litellm_model_response=_make_model_response(),
                model="vertex_ai/gemini-3.5-flash",
                kwargs={},
                start_time=datetime.now(),
                end_time=datetime.now(),
                logging_obj=logging_obj,
                custom_llm_provider="vertex_ai",
            )

        assert mock_cost.call_args.kwargs["router_model_id"] == "dep-gemini-35"

    def test_custom_pricing_flagged_when_resolved_id_has_prices(self):
        """The cost calculator only applies model_cost[router_model_id] pricing when
        custom_pricing is True. Passthrough litellm_params carry no pricing keys, so
        the helper must flag custom_pricing when the resolved id has a priced entry —
        otherwise the id is ignored and cost falls back to the list price."""
        import litellm

        logging_obj = MagicMock()
        logging_obj.get_router_model_id.return_value = None
        logging_obj.litellm_params = {}
        mock_router = MagicMock()
        mock_router.get_model_ids.return_value = ["dep-gem35"]

        with (
            patch("litellm.proxy.proxy_server.llm_router", mock_router),
            patch.dict(
                litellm.model_cost,
                {"dep-gem35": {"input_cost_per_token": 8.25e-07}},
                clear=False,
            ),
        ):
            kw = VertexPassthroughLoggingHandler._custom_pricing_kwargs(logging_obj, "vertex_ai/gemini-3.5-flash")

        assert kw["router_model_id"] == "dep-gem35"
        assert kw["custom_pricing"] is True

    def test_custom_pricing_not_flagged_when_resolved_id_has_no_prices(self):
        logging_obj = MagicMock()
        logging_obj.get_router_model_id.return_value = None
        logging_obj.litellm_params = {}
        mock_router = MagicMock()
        mock_router.get_model_ids.return_value = ["dep-nopricing"]

        with patch("litellm.proxy.proxy_server.llm_router", mock_router):
            kw = VertexPassthroughLoggingHandler._custom_pricing_kwargs(logging_obj, "vertex_ai/gemini-3.5-flash")

        assert kw["router_model_id"] == "dep-nopricing"
        assert kw["custom_pricing"] is False


class TestAllModelBearingCostCallsForwardCustomPricing:
    def test_every_model_bearing_completion_cost_forwards_custom_pricing(self):
        """Structural guard: every ``litellm.completion_cost(...)`` call in the
        Vertex passthrough handler that bills a real model must forward
        ``_custom_pricing_kwargs`` so custom pricing (model_info) is honored.

        This covers all current and future model-bearing paths at once
        (generateContent, predict, embedContent, create_video). If a refactor
        drops the spread from any one of them, this test fails. The synthetic
        search SKU (``vertex_ai/search_api``) has no per-deployment pricing and
        is intentionally exempt.
        """
        tree = ast.parse(inspect.getsource(VertexPassthroughLoggingHandler))

        checked = 0
        missing = []
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "completion_cost"
            ):
                continue

            model_kw = next(
                (kw.value.value for kw in node.keywords if kw.arg == "model" and isinstance(kw.value, ast.Constant)),
                None,
            )
            if model_kw == "vertex_ai/search_api":
                continue

            checked += 1
            forwards_helper = any(
                kw.arg is None
                and isinstance(kw.value, ast.Call)
                and isinstance(kw.value.func, ast.Attribute)
                and kw.value.func.attr == "_custom_pricing_kwargs"
                for kw in node.keywords
            )
            if not forwards_helper:
                missing.append(node.lineno)

        assert checked >= 4, (
            "expected at least 4 model-bearing completion_cost calls "
            f"(generateContent, predict, embedContent, create_video), found {checked}"
        )
        assert not missing, f"completion_cost calls missing **_custom_pricing_kwargs at class-relative lines {missing}"
