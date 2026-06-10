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
        # Lower the server-side minimum so this test can run sub-second.
        # Production deployments enforce a 1.0s floor (see
        # ``_KEEPALIVE_MIN_SECONDS``).
        patch.object(proxy_server_module, "_KEEPALIVE_MIN_SECONDS", 0.05),
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
    original_create_task = asyncio.create_task

    def _spy_create_task(coro, *args, **kwargs):
        task = original_create_task(coro, *args, **kwargs)
        created_tasks.append(task)
        return task

    async def _never_yields():
        # ``__anext__()`` on this generator will hang forever; the keepalive
        # wrapper's pending Task therefore never completes naturally.
        await asyncio.sleep(60)
        yield "unreachable"  # pragma: no cover

    async def _run_test():
        with patch.object(asyncio, "create_task", _spy_create_task):
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


def test_keepalive_seconds_below_minimum_is_clamped_up():
    """A hostile/buggy client could send ``keepalive_seconds=1e-9`` to force
    ``_iter_with_keepalive`` into a tight ``: ping`` busy-loop on any stalled
    stream (denial-of-service). The server must clamp such values up to
    ``_KEEPALIVE_MIN_SECONDS`` so the heartbeat rate stays bounded.

    Verified by passing a tiny ``keepalive_seconds``, stalling the upstream
    for less than the clamped floor, and asserting NO pings are emitted —
    if the un-clamped value had been honoured, hundreds of pings would
    appear during the stall window."""
    from litellm.proxy import proxy_server as proxy_server_module

    # 0.05s stall, but we'll set the floor to 0.5s. If the unclamped 1e-9
    # interval were honoured we'd see ~50,000,000 pings; if clamping works,
    # we see zero (no full keepalive interval elapses before the stall ends).
    upstream = _slow_chunk_stream(
        chunks=["first", "second"],
        stall_before_index=1,
        stall_seconds=0.05,
    )
    request_data = _make_request_data(keepalive_seconds=1e-9)
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
        # Set the floor explicitly so the test is self-contained and not
        # coupled to whatever the production default happens to be.
        patch.object(proxy_server_module, "_KEEPALIVE_MIN_SECONDS", 0.5),
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
    assert pings == [], (
        f"keepalive_seconds=1e-9 must be clamped up to the server-side "
        f"minimum; un-clamped, the 0.05s stall would emit a flood of pings. "
        f"got {len(pings)} pings: {emitted!r}"
    )
    data_lines = [e for e in emitted if isinstance(e, str) and e.startswith("data: ")]
    assert data_lines == ["data: first\n\n", "data: second\n\n", "data: [DONE]\n\n"]


def _make_deployment_obj(keepalive_seconds):
    """Stand-in for ``router.get_deployment(model_id=...)`` return value: an
    object with a ``litellm_params`` attribute that itself exposes the custom
    ``keepalive_seconds`` field via ``getattr`` (matching the real pydantic
    ``extra="allow"`` shape)."""
    params = MagicMock(name="litellm_params")
    # ``getattr(params, "keepalive_seconds", None)`` on a MagicMock returns
    # another MagicMock by default — force the attribute explicitly so the
    # ``getattr`` lookup returns the value we want (incl. None).
    params.keepalive_seconds = keepalive_seconds
    deployment = MagicMock(name="deployment")
    deployment.litellm_params = params
    return deployment


def test_resolve_keepalive_request_value_wins_over_deployment_default():
    """When the request supplies ``keepalive_seconds``, the deployment-level
    default is never consulted — request always overrides config."""
    from litellm.proxy import proxy_server as proxy_server_module

    fake_router = MagicMock(name="llm_router")
    fake_router.get_deployment.return_value = _make_deployment_obj(60.0)
    fake_router.get_model_list.return_value = [
        {"litellm_params": {"keepalive_seconds": 60.0}}
    ]

    with patch.object(proxy_server_module, "llm_router", fake_router):
        # Request value (5.0) wins over both deployment paths (60.0).
        resolved = proxy_server_module._resolve_keepalive_seconds(
            request_data={"model": "gpt-3.5-turbo", "keepalive_seconds": 5.0},
            response=MagicMock(_hidden_params={"model_id": "dep-id-1"}),
        )

    assert resolved == 5.0
    # Deployment lookup must not have been consulted at all.
    fake_router.get_deployment.assert_not_called()
    fake_router.get_model_list.assert_not_called()


def test_resolve_keepalive_falls_back_to_deployment_via_model_id():
    """When the request has no ``keepalive_seconds`` and the response carries a
    ``model_id`` in ``_hidden_params``, the resolver looks up that exact
    deployment via ``router.get_deployment(model_id=...)`` (the O(1) path)."""
    from litellm.proxy import proxy_server as proxy_server_module

    fake_router = MagicMock(name="llm_router")
    fake_router.get_deployment.return_value = _make_deployment_obj(42.0)
    # If this is touched we know the fallback path was taken when it
    # shouldn't have been.
    fake_router.get_model_list.return_value = [
        {"litellm_params": {"keepalive_seconds": 999.0}}
    ]

    with patch.object(proxy_server_module, "llm_router", fake_router):
        resolved = proxy_server_module._resolve_keepalive_seconds(
            request_data={"model": "gpt-3.5-turbo"},
            response=MagicMock(_hidden_params={"model_id": "dep-id-1"}),
        )

    assert resolved == 42.0
    fake_router.get_deployment.assert_called_once_with(model_id="dep-id-1")
    # The model-name fallback path is only used when ``get_deployment`` fails
    # to resolve.
    fake_router.get_model_list.assert_not_called()


def test_resolve_keepalive_falls_back_to_model_name_when_no_model_id():
    """When the response has no ``_hidden_params["model_id"]`` (some streaming
    response types don't carry it), the resolver falls back to
    ``router.get_model_list(model_name=...)`` — alias/wildcard/team aware."""
    from litellm.proxy import proxy_server as proxy_server_module

    fake_router = MagicMock(name="llm_router")
    fake_router.get_model_list.return_value = [
        {"litellm_params": {"keepalive_seconds": 30.0}}
    ]

    with patch.object(proxy_server_module, "llm_router", fake_router):
        resolved = proxy_server_module._resolve_keepalive_seconds(
            request_data={"model": "gpt-3.5-turbo"},
            response=MagicMock(spec=[]),  # no _hidden_params attribute
        )

    assert resolved == 30.0
    fake_router.get_model_list.assert_called_once_with(model_name="gpt-3.5-turbo")


def test_resolve_keepalive_zero_in_request_disables_even_with_deployment_default():
    """An explicit ``keepalive_seconds=0`` in the request disables the heartbeat
    even when the deployment has a non-zero default — the request always wins,
    including for the "disable" case."""
    from litellm.proxy import proxy_server as proxy_server_module

    fake_router = MagicMock(name="llm_router")
    fake_router.get_deployment.return_value = _make_deployment_obj(60.0)

    with patch.object(proxy_server_module, "llm_router", fake_router):
        resolved = proxy_server_module._resolve_keepalive_seconds(
            request_data={"model": "gpt-3.5-turbo", "keepalive_seconds": 0},
            response=MagicMock(_hidden_params={"model_id": "dep-id-1"}),
        )

    assert resolved == 0.0
    # Deployment must not be consulted — request 0 short-circuits.
    fake_router.get_deployment.assert_not_called()


def test_resolve_keepalive_returns_zero_when_router_is_none():
    """No router (e.g. proxy started without a config / model_list) — no
    fallback is possible, resolver returns 0 (disabled)."""
    from litellm.proxy import proxy_server as proxy_server_module

    with patch.object(proxy_server_module, "llm_router", None):
        resolved = proxy_server_module._resolve_keepalive_seconds(
            request_data={"model": "gpt-3.5-turbo"},
            response=MagicMock(),
        )

    assert resolved == 0.0


def test_resolve_keepalive_clamps_deployment_default_too():
    """The clamp must apply regardless of where the value came from — a
    deployment config with an out-of-band default still gets clamped."""
    from litellm.proxy import proxy_server as proxy_server_module

    fake_router = MagicMock(name="llm_router")
    fake_router.get_deployment.return_value = _make_deployment_obj(999999.0)

    with (
        patch.object(proxy_server_module, "llm_router", fake_router),
        patch.object(proxy_server_module, "_KEEPALIVE_MAX_SECONDS", 60.0),
    ):
        resolved = proxy_server_module._resolve_keepalive_seconds(
            request_data={"model": "gpt-3.5-turbo"},
            response=MagicMock(_hidden_params={"model_id": "dep-id-1"}),
        )

    assert resolved == 60.0


def test_async_data_generator_uses_deployment_config_keepalive():
    """End-to-end: with no ``keepalive_seconds`` on the request but a
    deployment-level default, ``async_data_generator`` emits ``: ping``
    heartbeats when the upstream stalls. Proves the resolver is wired
    into the streaming generator and not just unit-callable."""
    from litellm.proxy import proxy_server as proxy_server_module

    inner = _slow_chunk_stream(
        chunks=["first", "second"],
        stall_before_index=1,
        stall_seconds=0.3,
    )

    # Wrap the async generator in a class that also carries ``_hidden_params``
    # so the resolver's O(1) path (``router.get_deployment(model_id=...)``)
    # is exercised — raw async generators don't allow attribute assignment.
    class _UpstreamWithHiddenParams:
        def __init__(self, gen, hidden_params):
            self._gen = gen
            self._hidden_params = hidden_params

        def __aiter__(self):
            return self._gen.__aiter__()

    upstream = _UpstreamWithHiddenParams(inner, {"model_id": "dep-id-1"})

    fake_router = MagicMock(name="llm_router")
    fake_router.get_deployment.return_value = _make_deployment_obj(0.1)

    request_data = _make_request_data(keepalive_seconds=None)
    user_api_key_dict = MagicMock(name="user_api_key_dict")

    fake_logging = MagicMock(name="proxy_logging_obj")
    fake_logging.needs_iterator_wrap.return_value = False
    fake_logging.needs_per_chunk_streaming_hook.return_value = False

    with (
        patch.object(proxy_server_module, "proxy_logging_obj", fake_logging),
        patch.object(proxy_server_module, "llm_router", fake_router),
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
        # Lower the server-side minimum so this test can run sub-second
        # (same trick as ``test_keepalive_emits_ping_when_upstream_stalls``).
        patch.object(proxy_server_module, "_KEEPALIVE_MIN_SECONDS", 0.05),
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
    assert len(pings) >= 2, (
        f"deployment-level keepalive_seconds=0.1 should trigger heartbeats "
        f"during the 0.3s upstream stall; got {len(pings)} pings. full: {emitted!r}"
    )


def test_keepalive_seconds_above_maximum_is_clamped_down():
    """An interval longer than ``_KEEPALIVE_MAX_SECONDS`` would defeat the
    heartbeat (the intermediary proxy times out before our first ping).
    Verify that the server clamps such values down — exercising the
    other side of the clamp expression for branch coverage."""
    from litellm.proxy import proxy_server as proxy_server_module

    # Force the upper bound to a tiny value so the stall (0.15s) exceeds it
    # and we get a measurable number of pings if clamping worked. Without
    # the clamp, the request's 999999s interval would suppress every ping.
    upstream = _slow_chunk_stream(
        chunks=["first", "second"],
        stall_before_index=1,
        stall_seconds=0.15,
    )
    request_data = _make_request_data(keepalive_seconds=999999.0)
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
        patch.object(proxy_server_module, "_KEEPALIVE_MIN_SECONDS", 0.0),
        patch.object(proxy_server_module, "_KEEPALIVE_MAX_SECONDS", 0.05),
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
    assert len(pings) >= 1, (
        f"keepalive_seconds=999999 must be clamped down to the server-side "
        f"maximum; un-clamped, the 0.15s stall would emit zero pings. "
        f"got {len(pings)} pings: {emitted!r}"
    )
