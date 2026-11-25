import pytest

from litellm.integrations.opik.opik_payload_builder.extractors import (
    apply_proxy_header_overrides,
    extract_opik_metadata,
)


def test_extract_opik_metadata_fills_missing_keys_from_auth_metadata():
    litellm_metadata = {"opik": {"project_name": "my-proj"}}
    standard_logging_metadata = {
        "user_api_key_auth_metadata": {
            "opik": {
                "workspace": "auth-workspace",
                "project_name": "auth-project",
            }
        }
    }

    result = extract_opik_metadata(
        litellm_metadata=litellm_metadata,
        standard_logging_metadata=standard_logging_metadata,
    )

    assert result == {
        "project_name": "my-proj",
        "workspace": "auth-workspace",
    }


def test_extract_opik_metadata_request_metadata_overrides_auth_metadata():
    litellm_metadata = {
        "opik": {
            "workspace": "request-workspace",
            "thread_id": "request-thread",
        }
    }
    standard_logging_metadata = {
        "user_api_key_auth_metadata": {
            "opik": {
                "workspace": "auth-workspace",
                "thread_id": "auth-thread",
                "project_name": "auth-project",
            }
        }
    }

    result = extract_opik_metadata(
        litellm_metadata=litellm_metadata,
        standard_logging_metadata=standard_logging_metadata,
    )

    assert result == {
        "workspace": "request-workspace",
        "thread_id": "request-thread",
        "project_name": "auth-project",
    }


def test_extract_opik_metadata_requester_metadata_overrides_all_other_sources():
    litellm_metadata = {"opik": {"project_name": "request-project"}}
    standard_logging_metadata = {
        "user_api_key_auth_metadata": {
            "opik": {
                "workspace": "auth-workspace",
                "project_name": "auth-project",
            }
        },
        "requester_metadata": {
            "opik": {
                "workspace": "requester-workspace",
                "thread_id": "requester-thread",
                "project_name": "requester-project",
            }
        },
    }

    result = extract_opik_metadata(
        litellm_metadata=litellm_metadata,
        standard_logging_metadata=standard_logging_metadata,
    )

    assert result == {
        "project_name": "requester-project",
        "workspace": "requester-workspace",
        "thread_id": "requester-thread",
    }


@pytest.mark.parametrize(
    ("opik_tags_header", "expected_tags"),
    [
        ('["from-header", "second"]', ["from-request", "from-header", "second"]),
        ('["text", 7, null, {"nested": [1]}]', ["from-request", "text", 7, None, {"nested": [1]}]),
        ("[]", ["from-request"]),
        ('{"not": "a list"}', ["from-request"]),
        ('"not-a-list"', ["from-request"]),
        ("null", ["from-request"]),
        ("not json", ["from-request"]),
    ],
)
def test_opik_tags_header_adds_tags_only_when_it_is_a_json_list(opik_tags_header: str, expected_tags: list[object]):
    overrides = apply_proxy_header_overrides("project", ["from-request"], None, {"opik_tags": opik_tags_header})

    assert overrides == ("project", expected_tags, None)


def test_opik_headers_override_the_project_name_and_thread_id():
    overrides = apply_proxy_header_overrides(
        "project",
        ["from-request"],
        "thread-from-request",
        {"opik_project_name": "header-project", "opik_thread_id": "header-thread", "opik_tags": "", "x-other": "1"},
    )

    assert overrides == ("header-project", ["from-request"], "header-thread")
