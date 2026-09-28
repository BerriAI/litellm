import asyncio

import pytest


@pytest.mark.asyncio
async def test_user_config_router_stays_live_until_await_and_discards_after(monkeypatch):
    import litellm
    from litellm.proxy.route_llm_request import _route_user_config_request

    routers = []

    class FakeRouter:
        @staticmethod
        def get_valid_args():
            return ["model_list"]

        def __init__(self, **kwargs):
            self.discarded = False
            routers.append(self)

        async def acompletion(self, **kwargs):
            assert not self.discarded
            return "done"

        def discard(self):
            assert not self.discarded
            self.discarded = True

    monkeypatch.setattr(litellm, "Router", FakeRouter)
    call = _route_user_config_request({"user_config": {"model_list": []}}, "acompletion")
    assert not routers[0].discarded
    assert await call == "done"
    assert routers[0].discarded


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [ValueError("provider failed"), asyncio.CancelledError()])
async def test_user_config_router_discards_on_provider_error(monkeypatch, failure):
    import litellm
    from litellm.proxy.route_llm_request import _route_user_config_request

    routers = []

    class FakeRouter:
        @staticmethod
        def get_valid_args():
            return []

        def __init__(self):
            self.discarded = False
            routers.append(self)

        async def acompletion(self):
            assert not self.discarded
            raise failure

        def discard(self):
            self.discarded = True

    monkeypatch.setattr(litellm, "Router", FakeRouter)
    with pytest.raises(type(failure)):
        await _route_user_config_request({"user_config": {}}, "acompletion")
    assert routers[0].discarded


@pytest.mark.asyncio
async def test_user_config_stream_keeps_router_until_exhaustion_or_close(monkeypatch):
    import litellm
    from litellm.proxy.route_llm_request import _route_user_config_request

    routers = []

    class FakeStream:
        def __init__(self, router):
            self.router = router
            self.count = 0
            self.closed = False
            self._hidden_params = {"model_id": "test"}

        def __aiter__(self):
            return self

        async def __anext__(self):
            assert not self.router.discarded
            self.count += 1
            if self.count > 1:
                raise StopAsyncIteration
            return "chunk"

        async def aclose(self):
            assert not self.router.discarded
            self.closed = True

    class FakeRouter:
        @staticmethod
        def get_valid_args():
            return []

        def __init__(self):
            self.discarded = False
            self.discards = 0
            routers.append(self)

        async def acompletion(self, **kwargs):
            return FakeStream(self)

        def discard(self):
            self.discarded = True
            self.discards += 1

    monkeypatch.setattr(litellm, "Router", FakeRouter)
    for close_early in (False, True):
        stream = await _route_user_config_request({"user_config": {}, "stream": True}, "acompletion")
        router = routers[-1]
        assert not router.discarded
        assert stream._hidden_params == {"model_id": "test"}
        assert await anext(stream) == "chunk"
        assert not router.discarded
        if close_early:
            await stream.aclose()
            assert stream._stream.closed
        else:
            with pytest.raises(StopAsyncIteration):
                await anext(stream)
        assert router.discards == 1
        await stream.aclose()
        assert router.discards == 1


@pytest.mark.asyncio
async def test_user_config_stream_discards_when_iteration_raises(monkeypatch):
    import litellm
    from litellm.proxy.route_llm_request import _route_user_config_request

    routers = []

    class FakeRouter:
        @staticmethod
        def get_valid_args():
            return []

        def __init__(self):
            self.discards = 0
            routers.append(self)

        async def acompletion(self):
            async def stream():
                assert self.discards == 0
                raise ValueError("stream failed")
                yield "unreachable"

            return stream()

        def discard(self):
            self.discards += 1

    monkeypatch.setattr(litellm, "Router", FakeRouter)
    stream = await _route_user_config_request({"user_config": {}}, "acompletion")
    with pytest.raises(ValueError, match="stream failed"):
        await anext(stream)
    assert routers[0].discards == 1
