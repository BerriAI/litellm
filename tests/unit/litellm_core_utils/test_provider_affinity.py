import pytest

from litellm.constants import SESSION_ID_GENERATED_METADATA_KEY
from litellm.litellm_core_utils.provider_affinity import (
    add_provider_affinity_header,
    get_stable_session_id,
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

