from contextlib import asynccontextmanager

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy.common_utils.callback_lifecycle import close_proxy_callbacks, setup_proxy_callbacks


class RouteCallback(CustomLogger):
    def setup_proxy(self, app):
        @app.get("/callback-route")
        def callback_route():
            return {"source": "callback"}


def test_lifespan_hook_adds_route_and_refreshes_openapi():
    callback = RouteCallback()

    @asynccontextmanager
    async def lifespan(app):
        setup_proxy_callbacks(app, [callback, "unresolved-name", callback])
        yield

    app = FastAPI(lifespan=lifespan)
    assert "/callback-route" not in app.openapi()["paths"]
    with TestClient(app) as client:
        assert client.get("/callback-route").json() == {"source": "callback"}
        assert "/callback-route" in client.get("/openapi.json").json()["paths"]
    assert sum(route.path == "/callback-route" for route in app.routes) == 1


def test_failed_hook_aborts_lifespan_startup():
    class FailedCallback(CustomLogger):
        def setup_proxy(self, app):
            raise ValueError("invalid callback configuration")

    @asynccontextmanager
    async def lifespan(app):
        setup_proxy_callbacks(app, [FailedCallback()])
        yield

    with pytest.raises(ValueError, match="invalid callback configuration"):
        with TestClient(FastAPI(lifespan=lifespan)):
            pytest.fail("startup should not succeed")


@pytest.mark.asyncio
async def test_cleanup_closes_real_sessions_after_another_callback_fails():
    from litellm.proxy.guardrails.guardrail_hooks.presidio import OPTIONAL_PresidioPIIMasking

    class FailedCallback(CustomLogger):
        async def async_close(self):
            raise ValueError("cleanup failed")

    guardrail = OPTIONAL_PresidioPIIMasking(mock_testing=True)
    async with guardrail._get_session_iterator() as session:
        assert not session.closed
    await close_proxy_callbacks([FailedCallback(), guardrail, guardrail, "name"])
    assert session.closed
    assert guardrail._http_session is None
    await close_proxy_callbacks([guardrail])
    assert guardrail._http_session is None


@pytest.mark.asyncio
async def test_presidio_cleanup_uses_background_session_owner_loop():
    import asyncio
    import threading

    from litellm.proxy.guardrails.guardrail_hooks.presidio import OPTIONAL_PresidioPIIMasking

    guardrail = OPTIONAL_PresidioPIIMasking(mock_testing=True)
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()

    async def get_session():
        async with guardrail._get_session_iterator() as session:
            return session

    try:
        session = await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(get_session(), loop))
        assert not session.closed
        await guardrail.async_close()
        assert session.closed
        assert guardrail._loop_sessions == {}
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join()
        loop.close()
