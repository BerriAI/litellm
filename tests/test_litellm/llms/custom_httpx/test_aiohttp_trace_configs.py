"""
Test that aiohttp trace_configs are preserved when session is recreated.

Fixes https://github.com/BerriAI/litellm/issues/20174

The issue was that when a user provides a shared_session with trace_configs,
those trace_configs would be lost if the session needed to be recreated
(e.g., due to closed session or different event loop).
"""

import asyncio
import pytest
import aiohttp

from litellm.llms.custom_httpx.aiohttp_transport import LiteLLMAiohttpTransport


class TestAiohttpTraceConfigsPreservation:
    """Tests for trace_configs preservation in LiteLLMAiohttpTransport"""

    @pytest.mark.asyncio
    async def test_trace_configs_stored_from_client_session(self):
        """Test that trace_configs are extracted and stored from a ClientSession"""
        # Create a trace config
        trace_config = aiohttp.TraceConfig()
        
        # Track if our callback was invoked
        callback_invoked = False
        
        async def on_request_start(session, trace_config_ctx, params):
            nonlocal callback_invoked
            callback_invoked = True
        
        trace_config.on_request_start.append(on_request_start)
        
        # Create a ClientSession with trace_configs
        async with aiohttp.ClientSession(trace_configs=[trace_config]) as session:
            # Create transport with the session
            transport = LiteLLMAiohttpTransport(client=session)
            
            # Verify trace_configs were stored
            assert hasattr(transport, "_trace_configs")
            assert transport._trace_configs is not None
            assert len(transport._trace_configs) == 1
            assert transport._trace_configs[0] is trace_config

    @pytest.mark.asyncio
    async def test_trace_configs_used_when_creating_new_session(self):
        """Test that stored trace_configs are used when creating a new session"""
        # Create a trace config
        trace_config = aiohttp.TraceConfig()
        
        # Create a ClientSession with trace_configs
        session = aiohttp.ClientSession(trace_configs=[trace_config])
        
        # Create transport with the session
        transport = LiteLLMAiohttpTransport(client=session)
        
        # Close the session to force recreation
        await session.close()
        
        # This should create a new session with the stored trace_configs
        new_session = transport._create_session_with_trace_configs()
        
        try:
            # Verify the new session has trace_configs
            assert hasattr(new_session, "_trace_configs")
            assert new_session._trace_configs is not None
            assert len(new_session._trace_configs) == 1
        finally:
            await new_session.close()

    @pytest.mark.asyncio
    async def test_session_recreation_preserves_trace_configs(self):
        """Test that _get_valid_client_session preserves trace_configs"""
        # Create a trace config
        trace_config = aiohttp.TraceConfig()
        
        # Create a ClientSession with trace_configs
        session = aiohttp.ClientSession(trace_configs=[trace_config])
        
        # Create transport with the session
        transport = LiteLLMAiohttpTransport(client=session)
        
        # Close the session to force recreation
        await session.close()
        
        # Get a valid session (should recreate with trace_configs)
        new_session = transport._get_valid_client_session()
        
        try:
            # Verify the new session has trace_configs
            assert hasattr(new_session, "_trace_configs")
            assert new_session._trace_configs is not None
            assert len(new_session._trace_configs) == 1
        finally:
            await new_session.close()

    @pytest.mark.asyncio
    async def test_no_trace_configs_creates_plain_session(self):
        """Test that a plain ClientSession without trace_configs still works"""
        # Create a plain ClientSession without trace_configs
        session = aiohttp.ClientSession()
        
        # Create transport with the session
        transport = LiteLLMAiohttpTransport(client=session)
        
        # The _trace_configs should be empty list or None
        stored_configs = getattr(transport, "_trace_configs", None)
        assert stored_configs is None or len(stored_configs) == 0
        
        # Close the session to force recreation
        await session.close()
        
        # This should create a plain session
        new_session = transport._create_session_with_trace_configs()
        
        try:
            # Verify the new session was created (no error)
            assert new_session is not None
            assert not new_session.closed
        finally:
            await new_session.close()

    @pytest.mark.asyncio
    async def test_callable_factory_takes_precedence(self):
        """Test that a callable factory takes precedence over stored trace_configs"""
        # Create a trace config
        trace_config = aiohttp.TraceConfig()
        
        # Track factory calls
        factory_calls = 0
        
        def client_factory():
            nonlocal factory_calls
            factory_calls += 1
            return aiohttp.ClientSession()
        
        # Create transport with a factory
        transport = LiteLLMAiohttpTransport(client=client_factory)
        
        # Should not have _trace_configs when factory is used
        assert not hasattr(transport, "_trace_configs") or transport._trace_configs is None
        
        # Get a valid session (should use factory)
        session = transport._get_valid_client_session()
        
        try:
            assert factory_calls == 1
        finally:
            await session.close()
