import asyncio
import gc
import aiohttp
import pytest
from litellm.llms.custom_httpx.aiohttp_transport import LiteLLMAiohttpTransport


@pytest.mark.asyncio
async def test_litellm_aiohttp_transport_del_closes_session():
    """Verify that garbage collection of LiteLLMAiohttpTransport closes its owned ClientSession."""
    session = aiohttp.ClientSession()
    assert not session.closed

    transport = LiteLLMAiohttpTransport(client=session, owns_session=True)
    assert transport._owns_session is True

    # Drop transport reference and force GC sweep
    del transport
    gc.collect()
    await asyncio.sleep(0.05)  # Yield to event loop to allow scheduled close task to complete

    # The session must now be closed
    assert session.closed


if __name__ == "__main__":
    asyncio.run(test_litellm_aiohttp_transport_del_closes_session())
    print("G2_CHECK_SUCCESS")
