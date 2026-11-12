"""
Tests for Router interactions API endpoint initialization functions.
"""

from unittest.mock import AsyncMock

import pytest

import litellm
from litellm import Router


class TestInitializeInteractionsEndpoints:
    """Test cases for _initialize_interactions_endpoints method"""

    def test_initialize_interactions_endpoints_creates_methods(self):
        """Test that _initialize_interactions_endpoints creates the expected interaction methods on the router."""
        router = Router(
            model_list=[
                {
                    "model_name": "gpt-4",
                    "litellm_params": {"model": "gpt-4"},
                }
            ]
        )

        # Verify the interaction methods are created
        assert hasattr(router, "acreate_interaction")
        assert hasattr(router, "create_interaction")
        assert hasattr(router, "aget_interaction")
        assert hasattr(router, "get_interaction")
        assert hasattr(router, "adelete_interaction")
        assert hasattr(router, "delete_interaction")
        assert hasattr(router, "acancel_interaction")
        assert hasattr(router, "cancel_interaction")

        # Verify they are callable
        assert callable(router.acreate_interaction)
        assert callable(router.create_interaction)
        assert callable(router.aget_interaction)
        assert callable(router.get_interaction)


class TestInitializeManagedAgentsEndpoints:
    """Tests for _initialize_managed_agents_endpoints."""

    def test_initialize_managed_agents_endpoints_creates_methods(self):
        router = Router(
            model_list=[
                {
                    "model_name": "gpt-4",
                    "litellm_params": {"model": "gpt-4"},
                }
            ]
        )

        for method_name in (
            "acreate_agent",
            "alist_agents",
            "aget_agent",
            "adelete_agent",
            "alist_agent_versions",
        ):
            assert hasattr(router, method_name), f"missing {method_name}"
            assert callable(getattr(router, method_name)), f"{method_name} not callable"


@pytest.mark.parametrize(
    ("request_kwargs", "provider"),
    [({}, "gemini"), ({"custom_llm_provider": "vertex_ai"}, "vertex_ai")],
    ids=["defaults-to-gemini", "keeps-the-caller-provider"],
)
@pytest.mark.asyncio
async def test_interaction_endpoints_default_the_provider_to_gemini(
    monkeypatch: pytest.MonkeyPatch, request_kwargs: dict[str, str], provider: str
) -> None:
    provider_call = AsyncMock(return_value={"result": "success"})
    monkeypatch.setattr(litellm.interactions, "aget", provider_call)
    router = Router(model_list=[{"model_name": "gpt-4", "litellm_params": {"model": "gpt-4"}}])

    result = await router.aget_interaction(interaction_id="test-id", **request_kwargs)

    assert result == {"result": "success"}
    assert provider_call.call_args.kwargs["custom_llm_provider"] == provider
    assert provider_call.call_args.kwargs["interaction_id"] == "test-id"


@pytest.mark.asyncio
async def test_agent_only_interaction_does_not_send_the_agent_name_as_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Managed agent interactions must not pass agent name as model to the SDK."""
    provider_call = AsyncMock(return_value={"id": "int-1"})
    monkeypatch.setattr(litellm.interactions, "acreate", provider_call)
    router = Router(model_list=[{"model_name": "openai/*", "litellm_params": {"model": "gpt-4"}}])

    result = await router.acreate_interaction(
        agent="mqy-custom-slides-agent", model="mqy-custom-slides-agent", input="hello"
    )

    assert result == {"id": "int-1"}
    assert provider_call.call_args.kwargs["agent"] == "mqy-custom-slides-agent"
    assert provider_call.call_args.kwargs.get("model") is None
    assert provider_call.call_args.kwargs["custom_llm_provider"] == "gemini"


@pytest.mark.parametrize(
    ("request_kwargs", "provider"),
    [({}, "gemini"), ({"custom_llm_provider": "vertex_ai"}, "vertex_ai")],
    ids=["defaults-to-gemini", "keeps-the-caller-provider"],
)
@pytest.mark.asyncio
async def test_managed_agent_endpoints_default_the_provider_to_gemini(
    monkeypatch: pytest.MonkeyPatch, request_kwargs: dict[str, str], provider: str
) -> None:
    provider_call = AsyncMock(return_value={"agents": []})
    monkeypatch.setattr(litellm.interactions.agents, "alist", provider_call)
    router = Router(model_list=[])

    result = await router.alist_agents(**request_kwargs)

    assert result == {"agents": []}
    assert provider_call.call_args.kwargs["custom_llm_provider"] == provider
