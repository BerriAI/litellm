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


SAMPLING_REMOVED_MODEL: Final = "claude-test-sampling-removed"


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
    ("global_flag", "kwargs"),
    [
        (True, {}),
        (False, {"drop_params": "true"}),
        (False, {"drop_params": True}),
    ],
)
def test_body_params_drop_an_unsupported_param_under_either_drop_params_flag(
    monkeypatch: pytest.MonkeyPatch, global_flag: bool, kwargs: dict[str, object]
) -> None:
    monkeypatch.setattr(litellm, "drop_params", global_flag)
    monkeypatch.setitem(
        litellm.model_cost,
        SAMPLING_REMOVED_MODEL,
        {"litellm_provider": "anthropic", "mode": "chat", "supports_sampling_params": False},
    )

    body: Final = messages_host.body_params(
        {"model": f"anthropic/{SAMPLING_REMOVED_MODEL}", "max_tokens": 16, "temperature": 0.3, **kwargs}
    )

    assert body == {"max_tokens": 16}


@pytest.mark.parametrize("kwargs", [{}, {"drop_params": "nonsense"}, {"drop_params": False}])
def test_body_params_reject_an_unsupported_param_without_drop_params(
    monkeypatch: pytest.MonkeyPatch, kwargs: dict[str, object]
) -> None:
    monkeypatch.setattr(litellm, "drop_params", False)
    monkeypatch.setitem(
        litellm.model_cost,
        SAMPLING_REMOVED_MODEL,
        {"litellm_provider": "anthropic", "mode": "chat", "supports_sampling_params": False},
    )

    with pytest.raises(litellm.BadRequestError, match="temperature"):
        messages_host.body_params({"model": f"anthropic/{SAMPLING_REMOVED_MODEL}", "temperature": 0.3, **kwargs})


@pytest.mark.parametrize(
    ("model", "custom_llm_provider", "kept"),
    [
        ("anthropic/claude-sonnet-5", None, True),
        ("claude-sonnet-5", "anthropic", True),
        ("deepseek/deepseek-chat", None, False),
        ("vertex_ai/claude-sonnet-5", None, False),
    ],
)
def test_body_params_keep_service_tier_only_for_providers_that_accept_it(
    model: str, custom_llm_provider: str | None, kept: bool
) -> None:
    body: Final = messages_host.body_params(
        {"model": model, "custom_llm_provider": custom_llm_provider, "service_tier": "standard_only"}
    )

    assert ("service_tier" in body) is kept


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
