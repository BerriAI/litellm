"""Integration tests for protocol_routing with Router.

These tests verify the filter is correctly wired into Router's
async_get_healthy_deployments() filter chain.
"""
import pytest
from litellm import Router
from litellm.protocol_routing import (
    SupportedProtocol,
    ProtocolMismatchError,
    set_protocol_routing_mode,
    get_protocol_routing_mode,
)


@pytest.fixture
def mixed_deployments():
    """OpenAI + Anthropic deployments sharing the same model_name."""
    return [
        {
            "model_name": "shared-model",
            "litellm_params": {
                "model": "openai/gpt-4o",
                "custom_llm_provider": "openai",
                "api_key": "sk-fake",
            },
            "model_info": {"id": "openai-1"},
        },
        {
            "model_name": "shared-model",
            "litellm_params": {
                "model": "anthropic/claude-3-sonnet",
                "custom_llm_provider": "anthropic",
                "api_key": "sk-ant-fake",
            },
            "model_info": {"id": "anthropic-1"},
        },
    ]


@pytest.fixture(autouse=True)
def reset_mode():
    """Ensure each test starts with bridged mode and restores it after."""
    original = get_protocol_routing_mode()
    set_protocol_routing_mode("bridged")
    yield
    set_protocol_routing_mode(original)


class TestRouterIntegration:
    """Test protocol_routing filter wired into Router's filter chain."""

    @pytest.mark.asyncio
    async def test_bridged_mode_returns_both_deployments(self, mixed_deployments):
        """In bridged mode, both deployments are visible to Router."""
        router = Router(model_list=mixed_deployments)
        set_protocol_routing_mode("bridged")

        deployments = await router.async_get_healthy_deployments(
            model="shared-model",
            request_kwargs={"_route_type": "anthropic_messages"},
        )

        # Both deployments returned (bridged mode = no filter)
        assert isinstance(deployments, list)
        assert len(deployments) == 2

    @pytest.mark.asyncio
    async def test_strict_mode_filters_to_anthropic(self, mixed_deployments):
        """In strict mode, only Anthropic deployment passes for anthropic_messages."""
        router = Router(model_list=mixed_deployments)
        set_protocol_routing_mode("strict")

        deployments = await router.async_get_healthy_deployments(
            model="shared-model",
            request_kwargs={"_route_type": "anthropic_messages"},
        )

        assert isinstance(deployments, list)
        assert len(deployments) == 1
        assert deployments[0]["litellm_params"]["custom_llm_provider"] == "anthropic"

    @pytest.mark.asyncio
    async def test_strict_mode_filters_to_openai(self, mixed_deployments):
        """In strict mode, only OpenAI deployment passes for acompletion."""
        router = Router(model_list=mixed_deployments)
        set_protocol_routing_mode("strict")

        deployments = await router.async_get_healthy_deployments(
            model="shared-model",
            request_kwargs={"_route_type": "acompletion"},
        )

        assert isinstance(deployments, list)
        assert len(deployments) == 1
        assert deployments[0]["litellm_params"]["custom_llm_provider"] == "openai"

    @pytest.mark.asyncio
    async def test_strict_mode_raises_when_no_match(self, mixed_deployments):
        """In strict mode, ProtocolMismatchError raised when nothing matches."""
        router = Router(model_list=mixed_deployments)
        set_protocol_routing_mode("strict")

        with pytest.raises(ProtocolMismatchError) as exc_info:
            await router.async_get_healthy_deployments(
                model="shared-model",
                request_kwargs={"_route_type": "agenerate_content"},
            )

        err = exc_info.value
        assert err.requested_protocol == "google_generate_content"
        assert "openai_chat" in err.available_protocols
        assert "anthropic_messages" in err.available_protocols

    @pytest.mark.asyncio
    async def test_no_route_type_skips_filter(self, mixed_deployments):
        """Missing _route_type means no protocol filtering (legacy behavior)."""
        router = Router(model_list=mixed_deployments)
        set_protocol_routing_mode("strict")

        deployments = await router.async_get_healthy_deployments(
            model="shared-model",
            request_kwargs={},  # No _route_type
        )

        # All deployments returned because filter is skipped
        assert isinstance(deployments, list)
        assert len(deployments) == 2

    @pytest.mark.asyncio
    async def test_explicit_supported_protocols_respected(self):
        """Explicit model_info.supported_protocols overrides provider default."""
        deployments = [
            {
                "model_name": "shared-model",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                    "custom_llm_provider": "openai",
                    "api_key": "sk-fake",
                },
                "model_info": {
                    "id": "openai-with-anthropic-override",
                    "supported_protocols": ["anthropic_messages"],
                },
            },
        ]
        router = Router(model_list=deployments)
        set_protocol_routing_mode("strict")

        # OpenAI deployment with explicit anthropic_messages support
        result = await router.async_get_healthy_deployments(
            model="shared-model",
            request_kwargs={"_route_type": "anthropic_messages"},
        )

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["model_info"]["id"] == "openai-with-anthropic-override"

    @pytest.mark.asyncio
    async def test_passthrough_route_bypasses_filter(self, mixed_deployments):
        """allm_passthrough_route bypasses protocol filter even in strict mode."""
        router = Router(model_list=mixed_deployments)
        set_protocol_routing_mode("strict")

        deployments = await router.async_get_healthy_deployments(
            model="shared-model",
            request_kwargs={"_route_type": "allm_passthrough_route"},
        )

        # Both deployments returned (passthrough bypasses filter)
        assert isinstance(deployments, list)
        assert len(deployments) == 2


class TestHandlerStrictGuard:
    """Test the strict guard in anthropic messages handler."""

    @pytest.mark.asyncio
    async def test_strict_mode_blocks_openai_for_anthropic_messages(self):
        """Handler's strict guard rejects OpenAI provider for /v1/messages."""
        from litellm.llms.anthropic.experimental_pass_through.messages.handler import (
            anthropic_messages_handler,
        )

        set_protocol_routing_mode("strict")

        with pytest.raises(ProtocolMismatchError) as exc_info:
            anthropic_messages_handler(
                max_tokens=10,
                messages=[{"role": "user", "content": "hi"}],
                model="openai/gpt-4o",
                api_key="sk-fake",
            )

        err = exc_info.value
        assert err.requested_protocol == "anthropic_messages"
        assert "openai_chat" in err.available_protocols

    @pytest.mark.asyncio
    async def test_bridged_mode_allows_openai_for_anthropic_messages(self):
        """In bridged mode, handler allows OpenAI provider (with conversion)."""
        # We don't actually invoke the OpenAI API; just verify no
        # ProtocolMismatchError is raised before the dispatch.
        from litellm.llms.anthropic.experimental_pass_through.messages.handler import (
            anthropic_messages_handler,
        )

        set_protocol_routing_mode("bridged")

        # Will fail downstream (fake key), but NOT with ProtocolMismatchError
        try:
            anthropic_messages_handler(
                max_tokens=10,
                messages=[{"role": "user", "content": "hi"}],
                model="openai/gpt-4o",
                api_key="sk-fake-not-real",
            )
        except ProtocolMismatchError:
            pytest.fail("ProtocolMismatchError raised in bridged mode")
        except Exception:
            # Any other exception is acceptable (auth, network, etc.)
            pass
