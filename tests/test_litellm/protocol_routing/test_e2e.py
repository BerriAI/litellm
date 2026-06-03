"""End-to-end tests for protocol_routing through the proxy stack.

These tests exercise the full proxy dispatch path:
  route_request() -> Router.{call_type}() -> async_get_healthy_deployments()
                  -> filter_deployments_by_protocol()

without standing up an HTTP server. The TestClient approach was rejected
because it requires full proxy initialization (auth DB, lifespan, etc.)
which is too brittle for unit-style E2E coverage.
"""
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from litellm import Router
from litellm.protocol_routing import (
    ProtocolMismatchError,
    set_protocol_routing_mode,
    get_protocol_routing_mode,
)


@pytest.fixture
def mixed_router():
    """Router with OpenAI + Anthropic deployments sharing model_name."""
    return Router(
        model_list=[
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
    )


@pytest.fixture(autouse=True)
def reset_mode():
    original = get_protocol_routing_mode()
    set_protocol_routing_mode("bridged")
    yield
    set_protocol_routing_mode(original)


class TestRouteRequestPropagation:
    """Verify route_request() injects _route_type into data."""

    @pytest.mark.asyncio
    async def test_route_request_injects_route_type_into_data(self, mixed_router):
        """route_request() mutates data to add _route_type before dispatch."""
        from litellm.proxy.route_llm_request import route_request

        set_protocol_routing_mode("bridged")  # Don't filter, just check propagation

        data = {
            "model": "shared-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10,
        }

        # We don't care about the actual API call; any failure is fine.
        # The contract being tested: data["_route_type"] gets set by route_request.
        try:
            result = await route_request(
                data=data,
                llm_router=mixed_router,
                user_model=None,
                route_type="anthropic_messages",
            )
            # Router returns a coroutine; await it
            if hasattr(result, "__await__"):
                try:
                    await result
                except Exception:
                    pass
        except Exception:
            pass

        # The key contract: data was mutated with _route_type
        assert data.get("_route_type") == "anthropic_messages"

    @pytest.mark.asyncio
    async def test_end_to_end_strict_blocks_via_proxy_dispatch(self, mixed_router):
        """Full proxy dispatch path raises ProtocolMismatchError in strict mode.

        Verifies route_request -> Router.anthropic_messages ->
        async_get_healthy_deployments -> filter_deployments_by_protocol chain.
        """
        from litellm.proxy.route_llm_request import route_request

        # Router with only OpenAI deployments
        openai_only_router = Router(
            model_list=[
                {
                    "model_name": "gpt-only",
                    "litellm_params": {
                        "model": "openai/gpt-4o",
                        "custom_llm_provider": "openai",
                        "api_key": "sk-fake",
                    },
                },
            ]
        )

        set_protocol_routing_mode("strict")

        data = {
            "model": "gpt-only",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10,
        }

        with pytest.raises(ProtocolMismatchError):
            result = await route_request(
                data=data,
                llm_router=openai_only_router,
                user_model=None,
                route_type="anthropic_messages",
            )
            # Router method returns a coroutine; awaiting triggers the dispatch
            if hasattr(result, "__await__"):
                await result


class TestEndToEndStrictBlocking:
    """Strict mode end-to-end: request reaches Router and gets blocked."""

    @pytest.mark.asyncio
    async def test_anthropic_request_to_openai_only_router_blocked(self):
        """Anthropic /v1/messages -> OpenAI-only model -> ProtocolMismatchError."""
        router = Router(
            model_list=[
                {
                    "model_name": "gpt-only",
                    "litellm_params": {
                        "model": "openai/gpt-4o",
                        "custom_llm_provider": "openai",
                        "api_key": "sk-fake",
                    },
                },
            ]
        )
        set_protocol_routing_mode("strict")

        # Simulate proxy's dispatch path
        with pytest.raises(ProtocolMismatchError) as exc_info:
            await router.async_get_healthy_deployments(
                model="gpt-only",
                request_kwargs={"_route_type": "anthropic_messages"},
            )

        assert exc_info.value.requested_protocol == "anthropic_messages"

    @pytest.mark.asyncio
    async def test_anthropic_request_with_anthropic_deployment_passes(self, mixed_router):
        """Anthropic /v1/messages -> Anthropic deployment -> filtered correctly."""
        set_protocol_routing_mode("strict")

        deployments = await mixed_router.async_get_healthy_deployments(
            model="shared-model",
            request_kwargs={"_route_type": "anthropic_messages"},
        )

        # Only Anthropic deployment survives the filter
        assert isinstance(deployments, list)
        assert len(deployments) == 1
        assert deployments[0]["litellm_params"]["custom_llm_provider"] == "anthropic"


class TestEndToEndBridgedAllows:
    """Bridged mode end-to-end: cross-protocol conversion proceeds."""

    @pytest.mark.asyncio
    async def test_anthropic_to_openai_in_bridged_returns_both(self, mixed_router):
        """In bridged mode, both deployments visible (legacy conversion behavior)."""
        set_protocol_routing_mode("bridged")

        deployments = await mixed_router.async_get_healthy_deployments(
            model="shared-model",
            request_kwargs={"_route_type": "anthropic_messages"},
        )

        # Bridged mode returns ALL deployments — Router's strategy picks one
        assert isinstance(deployments, list)
        assert len(deployments) == 2


class TestEndToEndConfigOverride:
    """Explicit model_info.supported_protocols overrides provider defaults."""

    @pytest.mark.asyncio
    async def test_explicit_protocol_override_in_strict_mode(self):
        """Deployment with explicit supported_protocols passes strict check."""
        # OpenAI deployment but explicitly tagged as supporting anthropic_messages
        # (use case: OpenAI-compatible endpoint that speaks Anthropic protocol)
        router = Router(
            model_list=[
                {
                    "model_name": "claude-via-openai",
                    "litellm_params": {
                        "model": "openai/some-custom-anthropic-compatible",
                        "custom_llm_provider": "openai",
                        "api_key": "sk-fake",
                        "api_base": "https://example.com/anthropic-compat",
                    },
                    "model_info": {
                        "id": "custom-1",
                        "supported_protocols": ["anthropic_messages"],
                    },
                },
            ]
        )
        set_protocol_routing_mode("strict")

        deployments = await router.async_get_healthy_deployments(
            model="claude-via-openai",
            request_kwargs={"_route_type": "anthropic_messages"},
        )

        # Passes because explicit override declares anthropic_messages support
        assert isinstance(deployments, list)
        assert len(deployments) == 1


class TestHandlerStrictGuardE2E:
    """Verify handler's strict guard intercepts before adapter dispatch."""

    def test_strict_mode_blocks_openai_anthropic_messages_at_handler(self):
        """Direct call to anthropic_messages_handler() with OpenAI provider blocks in strict."""
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

    def test_bridged_mode_passes_handler_guard(self):
        """In bridged mode, handler guard does not block — call proceeds to adapter."""
        from litellm.llms.anthropic.experimental_pass_through.messages.handler import (
            anthropic_messages_handler,
        )

        set_protocol_routing_mode("bridged")

        # Will fail downstream (no real key), but NOT with ProtocolMismatchError
        try:
            anthropic_messages_handler(
                max_tokens=10,
                messages=[{"role": "user", "content": "hi"}],
                model="openai/gpt-4o",
                api_key="sk-fake-not-real",
            )
        except ProtocolMismatchError:
            pytest.fail("ProtocolMismatchError raised in bridged mode (regression)")
        except Exception:
            # Any other exception is fine (auth/network/etc.)
            pass

    def test_strict_mode_passes_for_native_anthropic_provider(self):
        """Native anthropic provider has provider_config — guard not even reached."""
        from litellm.llms.anthropic.experimental_pass_through.messages.handler import (
            anthropic_messages_handler,
        )

        set_protocol_routing_mode("strict")

        # Anthropic provider has BaseAnthropicMessagesConfig — bypasses our guard branch
        try:
            anthropic_messages_handler(
                max_tokens=10,
                messages=[{"role": "user", "content": "hi"}],
                model="anthropic/claude-3-sonnet",
                api_key="sk-ant-fake-not-real",
            )
        except ProtocolMismatchError:
            pytest.fail("ProtocolMismatchError raised for native Anthropic provider")
        except Exception:
            # Auth/network failure is fine
            pass
