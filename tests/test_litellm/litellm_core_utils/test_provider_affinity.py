from datetime import datetime
from unittest.mock import patch

import pytest

from litellm.constants import SESSION_ID_GENERATED_METADATA_KEY
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.provider_affinity import (
    PROVIDER_AFFINITY_REDACTED_VALUE,
    add_provider_affinity_header,
    get_stable_session_id,
    redact_provider_affinity_header,
)


@pytest.mark.parametrize(
    ("litellm_params", "expected"),
    [
        ({"litellm_session_id": "litellm-session"}, "litellm-session"),
        ({"session_id": "direct-session"}, "direct-session"),
        ({"metadata": {"session_id": "metadata-session"}}, "metadata-session"),
        ({"litellm_metadata": {"session_id": "litellm-metadata-session"}}, "litellm-metadata-session"),
    ],
)
def test_get_stable_session_id_uses_explicit_session_sources(litellm_params: dict, expected: str):
    assert get_stable_session_id(litellm_params) == expected


def test_get_stable_session_id_does_not_use_trace_id():
    assert get_stable_session_id({"litellm_trace_id": "per-request-trace"}) is None


@pytest.mark.parametrize("metadata_key", ["metadata", "litellm_metadata"])
def test_get_stable_session_id_ignores_proxy_generated_session(metadata_key: str):
    assert (
        get_stable_session_id(
            {
                "litellm_session_id": "generated-session",
                metadata_key: {
                    "session_id": "generated-session",
                    SESSION_ID_GENERATED_METADATA_KEY: True,
                },
            }
        )
        is None
    )


def test_get_stable_session_id_prefers_explicit_session_over_proxy_generated_session():
    assert (
        get_stable_session_id(
            {
                "session_id": "explicit-session",
                "litellm_session_id": "generated-session",
                "metadata": {
                    "session_id": "generated-session",
                    SESSION_ID_GENERATED_METADATA_KEY: True,
                },
            }
        )
        == "explicit-session"
    )


def test_add_provider_affinity_header_maps_session_id():
    headers = add_provider_affinity_header(
        headers={"Content-Type": "application/json"},
        litellm_params={
            "litellm_session_id": "session-123",
            "provider_affinity_header": "X-Conversation-Id",
        },
    )

    assert headers == {
        "Content-Type": "application/json",
        "X-Conversation-Id": "session-123",
    }


def test_add_provider_affinity_header_preserves_explicit_header_case_insensitively():
    headers = add_provider_affinity_header(
        headers={"x-conversation-id": "explicit-session"},
        litellm_params={
            "litellm_session_id": "session-123",
            "provider_affinity_header": "X-Conversation-Id",
        },
    )

    assert headers == {"x-conversation-id": "explicit-session"}


@pytest.mark.parametrize("session_id", ["session\r", "session\n", "session\0"])
def test_add_provider_affinity_header_rejects_control_characters(session_id: str):
    with pytest.raises(ValueError, match="session_id cannot contain HTTP header control characters"):
        add_provider_affinity_header(
            headers={},
            litellm_params={
                "litellm_session_id": session_id,
                "provider_affinity_header": "X-Conversation-Id",
            },
        )


def test_add_provider_affinity_header_does_nothing_without_config_or_session():
    assert add_provider_affinity_header({}, {"litellm_session_id": "session-123"}) == {}
    assert (
        add_provider_affinity_header(
            {},
            {"provider_affinity_header": "X-Conversation-Id"},
        )
        == {}
    )


def test_redact_provider_affinity_header_returns_a_copy_without_raw_value():
    headers = {"Authorization": "Bearer test", "X-Conversation-Id": "session-123"}

    redacted = redact_provider_affinity_header(
        headers=headers,
        litellm_params={
            "provider_affinity_header": "X-Conversation-Id",
        },
    )

    assert headers["X-Conversation-Id"] == "session-123"
    assert redacted == {
        "Authorization": "Bearer test",
        "X-Conversation-Id": PROVIDER_AFFINITY_REDACTED_VALUE,
    }


def test_pre_call_redacts_provider_affinity_header_from_all_logs_without_mutating_request():
    logging = Logging(
        model="gpt-5.5",
        messages=[{"role": "user", "content": "hello"}],
        stream=False,
        call_type="completion",
        start_time=datetime.now(),
        litellm_call_id="call-123",
        function_id="function-123",
    )
    logging.model_call_details["litellm_params"]["provider_affinity_header"] = "X-Conversation-Id"
    logging.model_call_details["litellm_params"]["metadata"] = {"request": "test"}
    logging.log_raw_request_response = True
    additional_args = {
        "api_base": "https://example.com/v1/chat/completions",
        "headers": {
            "X-Conversation-Id": "session-header",
            "X-Customer-Header": "customer-value",
        },
        "complete_input_dict": {
            "extra_headers": {
                "x-conversation-id": "session-extra-header",
                "X-Customer-Header": "customer-value",
            }
        },
    }

    with patch.object(logging, "_print_llm_call_debugging_log") as debug_log:
        logging.pre_call(input="hello", api_key="test-key", additional_args=additional_args)

    logged_args = logging.model_call_details["additional_args"]
    assert logged_args["headers"] == {
        "X-Conversation-Id": PROVIDER_AFFINITY_REDACTED_VALUE,
        "X-Customer-Header": "customer-value",
    }
    assert logged_args["complete_input_dict"]["extra_headers"] == {
        "x-conversation-id": PROVIDER_AFFINITY_REDACTED_VALUE,
        "X-Customer-Header": "customer-value",
    }
    assert debug_log.call_args.kwargs["headers"]["X-Conversation-Id"] == PROVIDER_AFFINITY_REDACTED_VALUE
    raw_request = logging.model_call_details["raw_request_typed_dict"]
    assert raw_request["raw_request_headers"]["X-Conversation-Id"] == PROVIDER_AFFINITY_REDACTED_VALUE
    assert raw_request["raw_request_body"]["extra_headers"]["x-conversation-id"] == PROVIDER_AFFINITY_REDACTED_VALUE
    assert "session-header" not in logging.model_call_details["litellm_params"]["metadata"]["raw_request"]
    assert "session-extra-header" not in logging.model_call_details["litellm_params"]["metadata"]["raw_request"]
    assert additional_args["headers"]["X-Conversation-Id"] == "session-header"
    assert additional_args["complete_input_dict"]["extra_headers"]["x-conversation-id"] == "session-extra-header"
