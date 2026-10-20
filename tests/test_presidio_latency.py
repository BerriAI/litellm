import asyncio
import aiohttp
import pytest
from unittest.mock import MagicMock, patch
from litellm.proxy.guardrails.guardrail_hooks.presidio import (
    _OPTIONAL_PresidioPIIMasking,
)


@pytest.mark.asyncio
async def test_sanity_presidio_session_reuse_main_thread():
    """
    SANITY CHECK:
    Verify that Presidio guardrail reuses sessions in the main thread.
    This ensures we don't break existing session pooling functionality.
    """
    presidio = _OPTIONAL_PresidioPIIMasking(
        mock_testing=True,
        presidio_analyzer_api_base="http://mock-analyzer",
        presidio_anonymizer_api_base="http://mock-anonymizer",
    )

    session_creations = 0
    original_init = aiohttp.ClientSession.__init__

    def mocked_init(self, *args, **kwargs):
        nonlocal session_creations
        session_creations += 1
        original_init(self, *args, **kwargs)

    with patch.object(
        aiohttp.ClientSession, "__init__", side_effect=mocked_init, autospec=True
    ):
        for _ in range(10):
            async with presidio._get_session_iterator() as session:
                pass

        # Expected: Only 1 session created for all 10 calls.
        assert session_creations == 1

    await presidio._close_http_session()


@pytest.mark.asyncio
async def test_bug_presidio_session_explosion_background_thread_causes_latency():
    """
    BUG REPRODUCTION:
    Verify that background threads (like logging hooks) REUSE sessions.
    Previously, each call in a background loop created a NEW ephemeral session,
    leading to socket exhaustion and the reported 97s latency spike.
    """
    import threading

    presidio = _OPTIONAL_PresidioPIIMasking(
        mock_testing=True,
        presidio_analyzer_api_base="http://mock-analyzer",
        presidio_anonymizer_api_base="http://mock-anonymizer",
    )

    # Force the code to think it's in a background thread
    presidio._main_thread_id = threading.get_ident() + 1

    session_creations = 0
    original_init = aiohttp.ClientSession.__init__

    def mocked_init(self, *args, **kwargs):
        nonlocal session_creations
        session_creations += 1
        original_init(self, *args, **kwargs)

    with patch.object(
        aiohttp.ClientSession, "__init__", side_effect=mocked_init, autospec=True
    ):
        for _ in range(10):
            async with presidio._get_session_iterator() as session:
                pass

        # FIX VERIFICATION: Should now be 1 session (reused) instead of 10.
        assert session_creations == 1

    await presidio._close_http_session()


@pytest.mark.asyncio
async def test_session_lock_is_not_held_while_request_uses_session():
    """
    The shared session lock must only protect session creation.

    If it is held while callers use the session, every Presidio HTTP request in
    a proxy process is serialized even though async_pre_call_hook fans out with
    asyncio.gather.
    """

    class DummySession:
        closed = False

        async def close(self):
            self.closed = True

    presidio = _OPTIONAL_PresidioPIIMasking(
        mock_testing=True,
        presidio_analyzer_api_base="http://mock-analyzer",
        presidio_anonymizer_api_base="http://mock-anonymizer",
    )
    presidio._http_session = DummySession()

    entered = 0
    both_entered = asyncio.Event()

    async def use_session():
        nonlocal entered
        async with presidio._get_session_iterator() as session:
            assert session is presidio._http_session
            entered += 1
            if entered == 2:
                both_entered.set()
            await asyncio.wait_for(both_entered.wait(), timeout=1.0)

    await asyncio.gather(use_session(), use_session())

    await presidio._close_http_session()
