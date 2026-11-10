from types import MappingProxyType
from typing import Final

import pytest

import litellm
from litellm.constants import DEFAULT_REQUEST_TIMEOUT_SECONDS
from litellm.rust_bridge.host.routes import messages as messages_host
from litellm.rust_bridge.host.routes.messages import response


def test_response_is_a_detached_public_messages_dict() -> None:
    native: Final = MappingProxyType(
        {
            "id": "msg_native",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5",
            "content": [{"type": "text", "text": "native"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 2, "output_tokens": 3},
        }
    )

    built: Final = response(native)

    assert built == dict(native)
    assert isinstance(built, dict)
    built["_hidden_params"] = {"annotated": True}
    assert "_hidden_params" not in native


def test_settings_project_caller_configuration_without_resolving_a_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "drop_params", False)
    monkeypatch.setattr(litellm, "reasoning_auto_summary", True)
    monkeypatch.setattr(litellm, "request_timeout", 1800)
    monkeypatch.setattr(litellm, "request_timeout_explicitly_set", True)

    projected: Final = messages_host.settings({"drop_params": "true", "additional_drop_params": ["metadata.user_id"]})

    assert projected == {
        "drop_params": True,
        "reasoning_auto_summary": True,
        "additional_drop_params": ("metadata.user_id",),
        "request_timeout": 1800.0,
    }


@pytest.mark.parametrize(
    ("timeout", "explicitly_set", "expected"),
    [
        (DEFAULT_REQUEST_TIMEOUT_SECONDS, False, None),
        (DEFAULT_REQUEST_TIMEOUT_SECONDS, True, DEFAULT_REQUEST_TIMEOUT_SECONDS),
        (300, False, 300.0),
    ],
)
def test_request_timeout_carries_only_a_configured_global_timeout(
    monkeypatch: pytest.MonkeyPatch, timeout: float, explicitly_set: bool, expected: float | None
) -> None:
    monkeypatch.setattr(litellm, "request_timeout", timeout)
    monkeypatch.setattr(litellm, "request_timeout_explicitly_set", explicitly_set)

    assert messages_host.settings({})["request_timeout"] == expected


@pytest.mark.parametrize(
    ("global_flag", "kwargs", "expected"),
    [
        (False, {}, False),
        (True, {}, True),
        (False, {"drop_params": "true"}, True),
        (False, {"drop_params": "nonsense"}, False),
        (False, {"drop_params": False}, False),
    ],
)
def test_drop_params_merges_the_global_flag_with_the_request(
    monkeypatch: pytest.MonkeyPatch, global_flag: bool, kwargs: dict[str, object], expected: bool
) -> None:
    monkeypatch.setattr(litellm, "drop_params", global_flag)

    assert messages_host.settings(kwargs)["drop_params"] is expected


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        (["tools[*].input_examples", 3, "metadata.user_id"], ("tools[*].input_examples", "metadata.user_id")),
        ("tools", ()),
        (None, ()),
    ],
)
def test_additional_drop_params_keep_only_string_paths(configured: object, expected: tuple[str, ...]) -> None:
    settings: Final = messages_host.settings({"additional_drop_params": configured})

    assert settings["additional_drop_params"] == expected


def test_native_request_rejections_map_to_the_public_400() -> None:
    from types import MappingProxyType

    from litellm.rust_bridge.public_call import NativeCall

    request: Final = NativeCall(
        args=(),
        kwargs=MappingProxyType({}),
        base={
            "model": "anthropic/claude-sonnet-5",
            "messages": (),
            "max_tokens": 8,
            "stream": None,
            "api_key": None,
            "api_base": None,
            "custom_llm_provider": None,
            **MappingProxyType({}),
        },
    )
    rejected: Final = ValueError("claude-sonnet-5 does not support top_k=5")
    rejected.messages_request_error = True  # pyright: ignore[reportAttributeAccessIssue]  # marker the native host sets

    mapped: Final = messages_host.map_failure(rejected, request.resolved, "anthropic")

    assert isinstance(mapped, litellm.BadRequestError)
    assert mapped.status_code == 400
    assert "does not support top_k=5" in mapped.message
    assert mapped.model == "claude-sonnet-5"
    assert not isinstance(
        messages_host.map_failure(ValueError("plain"), request.resolved, "anthropic"), litellm.BadRequestError
    )


def test_stream_hidden_params_projects_upstream_headers_the_way_the_python_handler_does() -> None:
    hidden: Final = messages_host.stream_hidden_params(
        (("request-id", "req_upstream_123"), ("x-ratelimit-remaining-requests", "41"))
    )

    additional: Final = hidden["additional_headers"]
    assert isinstance(additional, dict)
    assert additional["llm_provider-request-id"] == "req_upstream_123"
    assert additional["x-ratelimit-remaining-requests"] == "41"
    assert additional["request-id"] == additional["llm_provider-request-id"]


_ADAPTIVE_OPUS: Final = "anthropic/claude-opus-4-7"


@pytest.mark.parametrize(
    ("model", "expected"),
    [("claude-sonnet-4-5", "anthropic"), ("gpt-4o", "openai"), ("not-a-known-model", None)],
)
def test_inferred_provider_is_the_one_the_python_handler_resolves(model: str, expected: str | None) -> None:
    assert messages_host.inferred_provider(model, None) == expected


@pytest.mark.parametrize(
    ("request_effort", "levels", "drop_params", "expected"),
    [
        ("max", ["low", "high"], False, "high"),
        ("high", ["low"], True, "low"),
    ],
)
def test_body_params_round_effort_to_the_levels_the_deployment_declares(
    request_effort: str, levels: list[str], drop_params: bool, expected: str
) -> None:
    request: Final[dict[str, object]] = {
        "model": _ADAPTIVE_OPUS,
        "messages": [],
        "max_tokens": 8,
        "output_config": {"effort": request_effort},
        "drop_params": drop_params,
        "model_info": {"id": "deployment-1", "reasoning_effort_levels": levels},
    }

    assert messages_host.body_params(request)["output_config"] == {"effort": expected}


def test_body_params_keep_the_requested_effort_without_deployment_levels() -> None:
    request: Final[dict[str, object]] = {
        "model": _ADAPTIVE_OPUS,
        "messages": [],
        "max_tokens": 8,
        "output_config": {"effort": "max"},
    }

    assert messages_host.body_params(request)["output_config"] == {"effort": "max"}


@pytest.mark.parametrize(
    ("additional_drop_params", "expected_keys"),
    [
        (["thinking"], {"max_tokens"}),
        (["top_k"], {"max_tokens", "thinking", "output_config"}),
    ],
)
def test_body_params_derive_no_effort_from_a_param_the_deployment_lists(
    additional_drop_params: list[str], expected_keys: set[str]
) -> None:
    request: Final[dict[str, object]] = {
        "model": _ADAPTIVE_OPUS,
        "messages": [],
        "max_tokens": 8192,
        "thinking": {"type": "enabled", "budget_tokens": 4096},
        "additional_drop_params": additional_drop_params,
        "model_info": {"id": "deployment-1", "reasoning_effort_levels": ["low", "medium", "high"]},
    }

    assert set(messages_host.body_params(request)).difference({"messages", "model"}) == expected_keys


def _marked(phase: str | None = None, *args: object) -> ValueError:
    error: Final = ValueError(*args)
    if phase is not None:
        error.messages_timeout = phase  # pyright: ignore[reportAttributeAccessIssue]  # marker the native host sets
    return error


def test_a_stalled_stream_body_reaches_the_caller_as_the_transport_timeout() -> None:
    import httpx

    mapped: Final = messages_host.map_failure(_marked("Body"), {"model": _ADAPTIVE_OPUS}, "anthropic")

    assert isinstance(mapped, httpx.ReadTimeout)
    assert str(mapped) == "Timeout on reading data from socket"


def test_a_native_read_timeout_maps_to_the_public_timeout_with_python_text() -> None:
    mapped: Final = messages_host.map_failure(_marked("Read"), {"model": _ADAPTIVE_OPUS, "timeout": 1.0}, "anthropic")

    assert isinstance(mapped, litellm.Timeout)
    assert "Read timed out. Timeout passed=1.0" in str(mapped)


@pytest.mark.parametrize(("phase", "billed"), (("Read", True), ("Connection", False)))
def test_a_native_timeout_is_billed_on_the_failure_row_as_python_bills_it(phase: str, billed: bool) -> None:
    from litellm.proxy.utils import _provider_billed_the_failure

    mapped: Final = messages_host.map_failure(_marked(phase), {"model": _ADAPTIVE_OPUS, "timeout": 1.0}, "anthropic")

    assert isinstance(mapped, litellm.Timeout)
    assert _provider_billed_the_failure(mapped) is billed


def test_an_upstream_status_maps_through_the_provider_error_class() -> None:
    upstream: Final = ValueError(429, "slow down")
    upstream.headers = [("retry-after", "3")]  # pyright: ignore[reportAttributeAccessIssue]  # set by the native host

    mapped: Final = messages_host.map_failure(upstream, {"model": _ADAPTIVE_OPUS}, "anthropic")

    assert isinstance(mapped, litellm.RateLimitError)
    assert "slow down" in str(mapped)
    assert mapped.__context__ is upstream


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("", {"content-type": "text/plain"}),
        ("gateway", {"content-type": "text/plain", "content-length": "7"}),
    ],
)
def test_upstream_failure_headers_match_the_python_clients_rebuilt_response(
    body: str, expected: dict[str, str]
) -> None:
    from litellm.litellm_core_utils.exception_mapping_utils import _get_response_headers

    upstream: Final = ValueError(504, body)
    upstream.headers = [  # pyright: ignore[reportAttributeAccessIssue]  # set by the native host
        ("content-type", "text/plain"),
        ("content-encoding", "gzip"),
        ("content-length", "42"),
    ]

    mapped: Final = messages_host.map_failure(upstream, {"model": _ADAPTIVE_OPUS}, "anthropic")

    assert isinstance(mapped, litellm.Timeout)
    assert dict(_get_response_headers(mapped) or {}) == expected


@pytest.mark.parametrize("status", [200, 500])
def test_upstream_failure_without_wire_headers_states_no_headers(status: int) -> None:
    from litellm.litellm_core_utils.exception_mapping_utils import _get_response_headers

    upstream: Final = ValueError(status, "not a messages response")
    upstream.headers = []  # pyright: ignore[reportAttributeAccessIssue]  # set by the native host

    mapped: Final = messages_host.map_failure(upstream, {"model": _ADAPTIVE_OPUS}, "anthropic")

    assert isinstance(mapped, Exception)
    assert not dict(_get_response_headers(mapped) or {})


def test_a_late_rejection_tells_the_logger_the_resolved_call_first() -> None:
    import datetime

    from litellm.litellm_core_utils.litellm_logging import Logging

    logger: Final = Logging(
        model=_ADAPTIVE_OPUS,
        messages=[],
        stream=False,
        call_type="anthropic_messages",
        start_time=datetime.datetime.now(),
        litellm_call_id="late",
        function_id="late",
    )
    rejected: Final = ValueError("budget_tokens must be a number")
    rejected.messages_late_rejection = True  # pyright: ignore[reportAttributeAccessIssue]  # marker the native host sets
    rejected.messages_request_error = True  # pyright: ignore[reportAttributeAccessIssue]  # marker the native host sets

    mapped: Final = messages_host.map_failure(
        rejected,
        {"model": _ADAPTIVE_OPUS, "messages": [], "max_tokens": 8, "litellm_logging_obj": logger},
        "anthropic",
    )

    assert isinstance(mapped, litellm.BadRequestError)
    assert logger.model_call_details["custom_llm_provider"] == "anthropic"
    assert logger.model_call_details["optional_params"] == {"max_tokens": 8}


@pytest.mark.parametrize(
    ("aiohttp", "cut", "error_class", "text"),
    (
        (
            True,
            (60, None, False),
            "ReadError",
            "Response payload is not completed: <TransferEncodingError: 400, "
            "message='Not enough data to satisfy transfer length header.'>",
        ),
        (
            True,
            (19, 100, True),
            "ReadError",
            "Response payload is not completed: <ContentLengthError: 400, "
            "message='Not enough data to satisfy content length header (received 19 of 100 bytes).'>. "
            "ConnectionResetError(104, 'Connection reset by peer')",
        ),
        (
            False,
            (60, None, False),
            "RemoteProtocolError",
            "peer closed connection without sending complete message body (incomplete chunked read)",
        ),
        (
            False,
            (19, 100, False),
            "RemoteProtocolError",
            "peer closed connection without sending complete message body (received 19 bytes, expected 100)",
        ),
        (False, (19, 100, True), "ReadError", ""),
    ),
)
def test_a_stream_body_cut_reaches_the_caller_as_the_transport_raised_it(
    monkeypatch: pytest.MonkeyPatch,
    aiohttp: bool,
    cut: tuple[int, int | None, bool],
    error_class: str,
    text: str,
) -> None:
    import httpx

    from litellm.rust_bridge.host import http_transport

    monkeypatch.setattr(http_transport, "uses_aiohttp_transport", lambda: aiohttp)
    native: Final = RuntimeError("upstream network error")
    native.messages_body_cut = cut  # pyright: ignore[reportAttributeAccessIssue]  # marker the native host sets

    mapped: Final = messages_host.map_failure(native, {"model": _ADAPTIVE_OPUS}, "anthropic")

    assert type(mapped) is getattr(httpx, error_class)
    assert str(mapped) == text


_TRANSPORT: Final = "error sending request: client error"


@pytest.mark.parametrize(
    ("aiohttp", "native", "text"),
    (
        (
            True,
            f"could not reach the provider: {_TRANSPORT} (Connect): tcp connect error: Connection refused (os error 111)",
            "Cannot connect to host 127.0.0.1:9 ssl:<ssl> [Connect call failed ('127.0.0.1', 9)]",
        ),
        (
            True,
            f"could not reach the provider: {_TRANSPORT} (Connect): dns error: error resolving DNS: "
            "failed to lookup address information: Name or service not known",
            "Cannot connect to host 127.0.0.1:9 ssl:<ssl> [Name or service not known]",
        ),
        (
            True,
            f"could not reach the provider: {_TRANSPORT} (Connect): tls handshake eof",
            "Cannot connect to host 127.0.0.1:9 ssl:<ssl> [None]",
        ),
        (
            True,
            f"upstream network error: {_TRANSPORT} (SendRequest): connection error: Connection reset by peer (os error 104)",
            "[Errno 104] Connection reset by peer",
        ),
        (
            True,
            f"upstream network error: {_TRANSPORT} (SendRequest): connection closed before message completed",
            "Server disconnected",
        ),
        (
            False,
            f"could not reach the provider: {_TRANSPORT} (Connect): tcp connect error: Connection refused (os error 111)",
            "All connection attempts failed",
        ),
        (
            False,
            f"could not reach the provider: {_TRANSPORT} (Connect): dns error: error resolving DNS: "
            "failed to lookup address information: Name or service not known",
            "[Errno -2] Name or service not known",
        ),
        (False, f"could not reach the provider: {_TRANSPORT} (Connect): tls handshake eof", ""),
        (
            False,
            f"upstream network error: {_TRANSPORT} (SendRequest): connection error: Connection reset by peer (os error 104)",
            "",
        ),
        (
            False,
            f"upstream network error: {_TRANSPORT} (SendRequest): connection closed before message completed",
            "Server disconnected without sending a response.",
        ),
    ),
)
def test_a_failure_before_any_response_maps_like_the_python_transport_error(
    monkeypatch: pytest.MonkeyPatch, aiohttp: bool, native: str, text: str
) -> None:
    from litellm.rust_bridge.host import http_transport

    monkeypatch.setattr(http_transport, "uses_aiohttp_transport", lambda: aiohttp)
    monkeypatch.setattr(messages_host, "get_ssl_configuration", lambda _ssl_verify: "<ssl>")

    mapped: Final = messages_host.map_failure(
        RuntimeError(native), {"model": _ADAPTIVE_OPUS, "api_base": "http://127.0.0.1:9"}, "anthropic"
    )

    assert type(mapped) is litellm.InternalServerError
    assert mapped.status_code == 500
    assert mapped.message == (
        f"litellm.InternalServerError: AnthropicException - {text}. Handle with `litellm.InternalServerError`."
    )
    assert "This looks like a bug in LiteLLM" not in str(mapped)
