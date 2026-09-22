import json
from typing import Final

import pytest
from pydantic import JsonValue

from litellm.proxy.management_endpoints.liteask.redaction import sanitize


def test_nested_credentials_are_removed_but_management_identifiers_survive() -> None:
    value: Final[JsonValue] = {
        "key_alias": "production", "key_name": "sk-...abcd", "key_id": "key-123", "token": "a" * 64,
        "key": "sk-generatedcredential123456789012345678901234567890",
        "metadata": {"provider": {"api_key": "b" * 64, "access_token": "opaque-token"}},
        "nested": [{"credentials": {"client_secret": "secret"}}],
        "spend": 12.5, "total_tokens": 123, "user_id": "user-123",
    }
    assert sanitize(value, response_operation="key_update") == {
        "key_alias": "production", "key_name": "sk-...abcd", "key_id": "key-123", "token": "a" * 64,
        "key": "REDACTED", "metadata": {"provider": {"api_key": "REDACTED", "access_token": "REDACTED"}},
        "nested": [{"credentials": "REDACTED"}], "spend": 12.5, "total_tokens": 123, "user_id": "user-123",
    }


@pytest.mark.parametrize(
    ("operation", "container", "field"),
    (("keys_list", "keys", "token"), ("request_logs", "data", "api_key")),
)
def test_hash_identifiers_require_exact_response_provenance(operation: str, container: str, field: str) -> None:
    key_hash: Final = "a" * 64
    secret: Final = "b" * 64
    value: Final[JsonValue] = {
        container: [{field: key_hash, "metadata": {field: secret, "provider": {"api_key": secret}}}],
        "metadata": {container: [{field: secret}]},
    }
    output: Final = json.dumps(sanitize(value, response_operation=operation))
    assert key_hash in output
    assert secret not in output
    assert key_hash not in json.dumps(sanitize(value))
    assert key_hash not in json.dumps(sanitize(value, response_operation="request_log_detail"))
    assert key_hash not in json.dumps(sanitize(value, (key_hash,), response_operation=operation))


def test_credentials_inside_log_and_message_text_are_not_model_context() -> None:
    current: Final = "custom-session-credential"
    returned: Final = "sk-generatedcredential123456789012345678901234567890"
    value: Final[JsonValue] = {
        "messages": [{"role": "user", "content": f"Use {current} or {returned}"}],
        "error": "Authorization: Bearer opaquecredential1234567890",
        "response": '{"api_key":"custom-third-party-secret"}',
    }
    output: Final = json.dumps(sanitize(value, (current,)))
    assert current not in output
    assert returned not in output
    assert "opaquecredential1234567890" not in output
    assert "custom-third-party-secret" not in output
    assert "REDACTED" in output


def test_response_bounds_never_return_unredacted_overflow() -> None:
    value: Final[JsonValue] = {"rows": [{"text": "x" * 10_000, "api_key": "secret"} for _ in range(101)]}
    assert sanitize(value) == {"truncated": True, "message": "Result exceeds the size limit. Use pagination or a narrower query."}
    deep: JsonValue = {"api_key": "secret"}
    for _ in range(20):
        deep = {"nested": deep}
    encoded: Final = json.dumps(sanitize(deep))
    assert "secret" not in encoded
    assert "nested data omitted" in encoded


def test_generated_secret_is_not_allowed_under_safe_identifier_field() -> None:
    value: Final = "sk-generatedcredential123456789012345678901234567890"
    assert sanitize({"key_alias": value, "key_hash": value}) == {"key_alias": "REDACTED", "key_hash": "REDACTED"}


def test_known_secrets_are_removed_from_hash_fields_and_object_keys() -> None:
    value: Final = "a" * 64
    assert sanitize({value: "entry", "key": value}, (value,), response_operation="key_update") == {
        "REDACTED": "entry", "key": "REDACTED"
    }
