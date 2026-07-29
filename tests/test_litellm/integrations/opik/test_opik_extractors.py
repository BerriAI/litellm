from litellm.integrations.opik.opik_payload_builder.extractors import (
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
