from typing import Final

import pytest

from litellm.litellm_core_utils.classifier_logging import classifier_input_snapshot, masked_originating_request


@pytest.mark.parametrize("encoded", [False, True])
def test_classifier_snapshot_preserves_provider_shape_and_is_independent(encoded: bool) -> None:
    import json

    provider_body: Final = {"system": [{"text": "rubric"}], "messages": [{"role": "user", "content": "ask"}]}
    snapshot: Final = classifier_input_snapshot(json.dumps(provider_body) if encoded else provider_body)
    assert snapshot == provider_body
    provider_body["messages"][0]["content"] = "later mutation"
    assert snapshot == {"system": [{"text": "rubric"}], "messages": [{"role": "user", "content": "ask"}]}


def test_originating_snapshot_masks_nested_credentials_without_altering_source() -> None:
    body: Final = {
        "model": "router",
        "input": [{"type": "message", "role": "user", "content": "source-only"}],
        "api_key": "short",
        "metadata": {"nested": [{"Authorization": "Bearer secret", "access_token": 123}]},
    }
    snapshot: Final = masked_originating_request({"proxy_server_request": {"body": body}})
    assert snapshot is not None
    assert snapshot["model"] == "router"
    assert snapshot["input"] == body["input"]
    assert snapshot["api_key"] == "REDACTED"
    assert snapshot["metadata"] == {"nested": [{"Authorization": "REDACTED", "access_token": "REDACTED"}]}
    assert body["api_key"] == "short"
    assert body["metadata"]["nested"][0]["Authorization"] == "Bearer secret"


@pytest.mark.parametrize("header", ["Cookie", "cookie", "COOKIE", "sEt-CoOkIe"])
def test_originating_snapshot_redacts_cookie_headers_shared_with_caller_metadata(header: str) -> None:
    headers: Final = {header: "session=synthetic-session-credential", "content-type": "application/json"}
    body: Final = {"messages": [{"role": "user", "content": "hello"}], "metadata": {"headers": headers}}
    snapshot: Final = masked_originating_request({"proxy_server_request": {"body": body, "headers": headers}})
    assert snapshot == {
        "messages": [{"role": "user", "content": "hello"}],
        "metadata": {"headers": {header: "REDACTED", "content-type": "application/json"}},
    }
    assert headers[header] == "session=synthetic-session-credential"
    assert body["metadata"]["headers"][header] == "session=synthetic-session-credential"


@pytest.mark.parametrize("value", [None, "not-json", [], {"messages": object()}])
def test_invalid_provider_payload_is_not_reported_as_captured(value: object) -> None:
    assert classifier_input_snapshot(value) is None
