"""
Tests for client-controlled SSE keepalive in ``async_data_generator``.

When ``keepalive_seconds`` (> 0) is set on the request body, the proxy emits an
SSE comment (``: ping``) if no upstream chunk arrives within the interval. This
keeps the connection alive while the model is generating but producing chunks
the OpenAI translation layer filters out (e.g. Anthropic ``ping`` events) and
prevents idle-stream cuts by intermediary proxies (ALB, nginx, sandboxed
L7 inference proxies).

Absent or ``0`` -> no keepalive, no behaviour change vs. the upstream fast path.
"""

import asyncio
from unittest.mock import MagicMock, patch


def _run(coro):
    """Run ``coro`` on a fresh event loop and return its result.

    Avoids the pytest-asyncio plugin so this test file stands alone.
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def _collect(aiter):
    out = []
    async for item in aiter:
        out.append(item)
    return out


def _slow_chunk_stream(chunks, stall_before_index, stall_seconds):
    """Async generator yielding ``chunks``; before yielding ``chunks[stall_before_index]``
    it sleeps ``stall_seconds``. Lets tests force a measurable gap between
    upstream chunks so the keepalive interval can fire deterministically."""

    async def _gen():
        for i, c in enumerate(chunks):
            if i == stall_before_index:
                await asyncio.sleep(stall_seconds)
            yield c

    return _gen()


def _make_request_data(keepalive_seconds=None):
    data = {"model": "gpt-3.5-turbo", "stream": True}
    if keepalive_seconds is not None:
        data["keepalive_seconds"] = keepalive_seconds
    return data


def test_keepalive_emits_ping_when_upstream_stalls():
    """With ``keepalive_seconds=0.1`` and an upstream that stalls 0.3s before
    its second chunk, at least two ``: ping`` heartbeats should appear between
    the two real chunks. Real chunks and ``[DONE]`` are still delivered in
    order."""
    from litellm.proxy import proxy_server as proxy_server_module

    upstream = _slow_chunk_stream(
        chunks=["first", "second"],
        stall_before_index=1,
        stall_seconds=0.3,
    )

    request_data = _make_request_data(keepalive_seconds=0.1)
    user_api_key_dict = MagicMock(name="user_api_key_dict")

    fake_logging = MagicMock(name="proxy_logging_obj")
    fake_logging.needs_iterator_wrap.return_value = False
    fake_logging.needs_per_chunk_streaming_hook.return_value = False

    with (
        patch.object(proxy_server_module, "proxy_logging_obj", fake_logging),
        patch.object(
            proxy_server_module,
            "_get_client_requested_model_for_streaming",
            return_value=None,
        ),
        patch.object(
            proxy_server_module.ProxyLogging,
            "_fire_deferred_stream_logging",
            return_value=None,
        ),
    ):
        emitted = _run(
            _collect(
                proxy_server_module.async_data_generator(
                    response=upstream,
                    user_api_key_dict=user_api_key_dict,
                    request_data=request_data,
                )
            )
        )

    pings = [e for e in emitted if e == ": ping\n\n"]
    data_lines = [e for e in emitted if isinstance(e, str) and e.startswith("data: ")]

    assert len(pings) >= 2, (
        f"expected at least 2 ping heartbeats during 0.3s stall with 0.1s "
        f"keepalive; got {len(pings)} pings. full output: {emitted!r}"
    )
    # Real chunks still arrive in order, with [DONE] at the end.
    assert data_lines == [
        "data: first\n\n",
        "data: second\n\n",
        "data: [DONE]\n\n",
    ], f"chunks/DONE were re-ordered or lost: {data_lines!r}"
    # Pings must appear strictly between the first chunk and the [DONE] marker.
    first_chunk_idx = emitted.index("data: first\n\n")
    done_idx = emitted.index("data: [DONE]\n\n")
    for i, item in enumerate(emitted):
        if item == ": ping\n\n":
            assert first_chunk_idx < i < done_idx, (
                f"ping at index {i} should be between first chunk "
                f"({first_chunk_idx}) and [DONE] ({done_idx}); full: {emitted!r}"
            )


def test_no_keepalive_field_means_no_pings():
    """With ``keepalive_seconds`` absent from the request, the output must not
    contain any ``: ping`` heartbeats — preserves the upstream behaviour for
    callers that don't opt in."""
    from litellm.proxy import proxy_server as proxy_server_module

    # Same stall as the positive test — proves the gap alone doesn't emit pings
    # when keepalive isn't requested.
    upstream = _slow_chunk_stream(
        chunks=["first", "second"],
        stall_before_index=1,
        stall_seconds=0.2,
    )
    request_data = _make_request_data(keepalive_seconds=None)
    user_api_key_dict = MagicMock(name="user_api_key_dict")

    fake_logging = MagicMock(name="proxy_logging_obj")
    fake_logging.needs_iterator_wrap.return_value = False
    fake_logging.needs_per_chunk_streaming_hook.return_value = False

    with (
        patch.object(proxy_server_module, "proxy_logging_obj", fake_logging),
        patch.object(
            proxy_server_module,
            "_get_client_requested_model_for_streaming",
            return_value=None,
        ),
        patch.object(
            proxy_server_module.ProxyLogging,
            "_fire_deferred_stream_logging",
            return_value=None,
        ),
    ):
        emitted = _run(
            _collect(
                proxy_server_module.async_data_generator(
                    response=upstream,
                    user_api_key_dict=user_api_key_dict,
                    request_data=request_data,
                )
            )
        )

    assert (
        ": ping\n\n" not in emitted
    ), f"no ping should be emitted when keepalive_seconds is unset; got: {emitted!r}"
    data_lines = [e for e in emitted if isinstance(e, str) and e.startswith("data: ")]
    assert data_lines == ["data: first\n\n", "data: second\n\n", "data: [DONE]\n\n"]


def test_keepalive_with_non_numeric_value_is_treated_as_disabled():
    """A malformed ``keepalive_seconds`` (string, dict, etc.) must be treated as
    disabled rather than raising. Real chunks must still be delivered."""
    from litellm.proxy import proxy_server as proxy_server_module

    upstream = _slow_chunk_stream(
        chunks=["first", "second"],
        stall_before_index=1,
        stall_seconds=0.0,  # no stall — speed test up
    )
    request_data = _make_request_data(keepalive_seconds="not-a-number")
    user_api_key_dict = MagicMock(name="user_api_key_dict")

    fake_logging = MagicMock(name="proxy_logging_obj")
    fake_logging.needs_iterator_wrap.return_value = False
    fake_logging.needs_per_chunk_streaming_hook.return_value = False

    with (
        patch.object(proxy_server_module, "proxy_logging_obj", fake_logging),
        patch.object(
            proxy_server_module,
            "_get_client_requested_model_for_streaming",
            return_value=None,
        ),
        patch.object(
            proxy_server_module.ProxyLogging,
            "_fire_deferred_stream_logging",
            return_value=None,
        ),
    ):
        emitted = _run(
            _collect(
                proxy_server_module.async_data_generator(
                    response=upstream,
                    user_api_key_dict=user_api_key_dict,
                    request_data=request_data,
                )
            )
        )

    assert ": ping\n\n" not in emitted
    data_lines = [e for e in emitted if isinstance(e, str) and e.startswith("data: ")]
    assert data_lines == ["data: first\n\n", "data: second\n\n", "data: [DONE]\n\n"]


def test_keepalive_seconds_is_in_all_litellm_params():
    """``keepalive_seconds`` must be in the proxy-only allowlist so it's stripped
    from the request before being forwarded to the upstream provider. Without
    this, the field would leak to providers (e.g. OpenAI, Anthropic) that
    don't accept it."""
    from litellm.types.utils import all_litellm_params

    assert "keepalive_seconds" in all_litellm_params


def test_iter_with_keepalive_fast_path_when_disabled():
    """When ``keepalive_seconds <= 0`` the helper is a plain ``async for`` —
    no Task wrapping, no sentinel emission. This path is short-circuited by
    ``async_data_generator``'s outer guard, but is still reachable for any
    caller that uses the helper directly (and is the documented behaviour),
    so cover it explicitly."""
    from litellm.proxy import proxy_server as proxy_server_module

    async def _chunks():
        yield "a"
        yield "b"
        yield "c"

    async def _run_test():
        wrapper = proxy_server_module._iter_with_keepalive(
            _chunks().__aiter__(), keepalive_seconds=0
        )
        out = [item async for item in wrapper]
        assert out == ["a", "b", "c"]
        # The sentinel is never yielded on the fast path.
        assert proxy_server_module._STREAM_KEEPALIVE not in out

    _run(_run_test())


def test_iter_with_keepalive_cancels_pending_task_on_early_close():
    """When the keepalive-wrapped generator is closed while an upstream
    ``__anext__()`` Task is still in-flight (e.g. the client disconnects
    mid-stream), the pending Task must be cancelled in the ``finally``
    clause so it doesn't leak.

    Exercised by:
    1. wrapping an upstream that never yields,
    2. consuming a few keepalive sentinels (proving a Task is in-flight),
    3. closing the wrapper generator,
    4. asserting the underlying Task transitioned to ``cancelled()``.
    """
    from litellm.proxy import proxy_server as proxy_server_module

    # Track the Tasks ``_iter_with_keepalive`` creates so we can assert on
    # cancellation after the wrapper is closed.
    created_tasks = []
    original_ensure_future = asyncio.ensure_future

    def _spy_ensure_future(coro_or_future, *args, **kwargs):
        task = original_ensure_future(coro_or_future, *args, **kwargs)
        created_tasks.append(task)
        return task

    async def _never_yields():
        # ``__anext__()`` on this generator will hang forever; the keepalive
        # wrapper's pending Task therefore never completes naturally.
        await asyncio.sleep(60)
        yield "unreachable"  # pragma: no cover

    async def _run_test():
        with patch.object(asyncio, "ensure_future", _spy_ensure_future):
            wrapper = proxy_server_module._iter_with_keepalive(
                _never_yields().__aiter__(), keepalive_seconds=0.05
            )
            # Drain a couple of keepalive sentinels — proves the wrapper has
            # an in-flight Task pending on ``__anext__()`` at this point.
            sentinel = await asyncio.wait_for(wrapper.__anext__(), timeout=1.0)
            assert sentinel is proxy_server_module._STREAM_KEEPALIVE
            sentinel = await asyncio.wait_for(wrapper.__anext__(), timeout=1.0)
            assert sentinel is proxy_server_module._STREAM_KEEPALIVE

            # Close the wrapper while the underlying Task is still in-flight.
            await wrapper.aclose()

            # Yield control so the cancellation actually propagates to the
            # Task (cancel() schedules; the loop tick delivers).
            await asyncio.sleep(0)

        # The wrapper re-uses a single pending Task across keepalive sentinels
        # (only allocates a new one after a real chunk arrives), so exactly
        # one Task should have been created and it should be cancelled.
        assert (
            len(created_tasks) == 1
        ), f"expected exactly 1 in-flight Task; got {len(created_tasks)}"
        assert created_tasks[0].cancelled(), (
            f"the in-flight __anext__() Task should be cancelled after early "
            f"close; got task.done()={created_tasks[0].done()} "
            f"task.cancelled()={created_tasks[0].cancelled()}"
        )

    _run(_run_test())
