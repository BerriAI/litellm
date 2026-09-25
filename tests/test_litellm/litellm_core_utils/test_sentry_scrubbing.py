import hashlib
import json
from collections.abc import Mapping
from functools import reduce
from typing import Final, cast

import pytest
import sentry_sdk
from pydantic import JsonValue
from sentry_sdk.envelope import Envelope
from sentry_sdk.transport import Transport
from sentry_sdk.utils import event_from_exception

from litellm.litellm_core_utils.sentry_scrubbing import (
    FILTERED,
    MAX_SCRUB_DEPTH,
    build_sentry_init_options,
    build_string_scrubber,
    scrub_json_strings,
)
from litellm.proxy._types import LiteLLM_UserTable, UserAPIKeyAuth

EMAIL: Final = "qa.user@example.com"
VIRTUAL_KEY: Final = "sk-virtual-key-under-test"
KEY_HASH: Final = hashlib.sha256(VIRTUAL_KEY.encode()).hexdigest()
MASTER_KEY: Final = "sk-master-key-under-test"
DATABASE_URL: Final = "postgresql://litellm:db-password-under-test@db.internal:5432/litellm"
PII_ON: Final = {"SENTRY_DSN": "https://key@sentry.example/1", "SENTRY_SEND_DEFAULT_PII": "true"}
PII_OFF: Final = {"SENTRY_DSN": "https://key@sentry.example/1"}


class RecordingTransport(Transport):
    def __init__(self) -> None:
        super().__init__()
        self.last_envelope: Envelope | None = None

    def capture_envelope(self, envelope: Envelope) -> None:
        self.last_envelope = envelope


def reject_request(
    valid_token: UserAPIKeyAuth,
    user_obj: LiteLLM_UserTable,
    general_settings: Mapping[str, str],
    data: Mapping[str, Mapping[str, str]],
    raw_headers: Mapping[str, str],
) -> None:
    raise RuntimeError(f"key {valid_token.token} owned by {user_obj.user_email} was rejected")


def raise_with_identity_locals() -> None:
    reject_request(
        valid_token=UserAPIKeyAuth(token=KEY_HASH, key_name="sk-...test", user_id=EMAIL, user_email=EMAIL),
        user_obj=LiteLLM_UserTable(user_id=EMAIL, user_email=EMAIL, user_role="internal_user"),
        general_settings={"master_key": MASTER_KEY, "database_url": DATABASE_URL},
        data={"metadata": {"user_api_key_hash": KEY_HASH, "user_api_key_user_email": EMAIL}},
        raw_headers={"authorization": f"Bearer {VIRTUAL_KEY}", "x-api-key": VIRTUAL_KEY, "content-type": "application/json"},
    )


def capture_serialized_event(env: Mapping[str, str]) -> str:
    transport: Final = RecordingTransport()
    client: Final = sentry_sdk.Client(transport=transport, **build_sentry_init_options(env))
    try:
        raise_with_identity_locals()
    except RuntimeError as error:
        event, hint = event_from_exception(error, client_options=client.options)
        client.capture_event(event, hint=hint)
    assert transport.last_envelope is not None
    return json.dumps(transport.last_envelope.items[0].payload.json)


def innermost_frame_vars(serialized: str) -> dict[str, JsonValue]:
    event: Final = json.loads(serialized)
    frames: Final = event["exception"]["values"][0]["stacktrace"]["frames"]
    return frames[-1]["vars"]


def test_default_event_carries_no_email_hash_or_secret_anywhere() -> None:
    serialized: Final = capture_serialized_event(PII_OFF)
    assert EMAIL not in serialized
    assert KEY_HASH not in serialized
    assert MASTER_KEY not in serialized
    assert VIRTUAL_KEY not in serialized
    assert "db-password-under-test" not in serialized
    frame_vars: Final = innermost_frame_vars(serialized)
    assert frame_vars["raw_headers"] == {"authorization": FILTERED, "x-api-key": FILTERED, "content-type": "'application/json'"}
    assert f"token='{FILTERED}'" in frame_vars["valid_token"]
    assert f"user_id='{FILTERED}'" in frame_vars["valid_token"]
    assert f"user_email='{FILTERED}'" in frame_vars["user_obj"]
    assert frame_vars["general_settings"] == {"master_key": FILTERED, "database_url": FILTERED}
    assert frame_vars["data"] == {"metadata": {"user_api_key_hash": FILTERED, "user_api_key_user_email": FILTERED}}
    assert "key_name='sk-...test'" in frame_vars["valid_token"]
    assert "user_role='internal_user'" in frame_vars["user_obj"]


def test_source_context_lines_are_left_readable() -> None:
    frames: Final = json.loads(capture_serialized_event(PII_OFF))["exception"]["values"][0]["stacktrace"]["frames"]
    source_lines: Final = tuple(
        line
        for frame in frames
        for line in (*frame.get("pre_context", []), frame.get("context_line", ""), *frame.get("post_context", []))
    )
    assert any("token=KEY_HASH" in line for line in source_lines)
    assert not any(FILTERED in line for line in source_lines)


def test_default_event_keeps_the_exception_message_shape() -> None:
    serialized: Final = capture_serialized_event(PII_OFF)
    message: Final = json.loads(serialized)["exception"]["values"][0]["value"]
    assert message == f"key {FILTERED} owned by {FILTERED} was rejected"


def test_pii_opt_in_keeps_identifiers_and_still_scrubs_secrets() -> None:
    serialized: Final = capture_serialized_event(PII_ON)
    frame_vars: Final = innermost_frame_vars(serialized)
    assert f"user_id='{EMAIL}'" in frame_vars["valid_token"]
    assert f"user_email='{EMAIL}'" in frame_vars["user_obj"]
    assert frame_vars["data"] == {
        "metadata": {"user_api_key_hash": f"'{KEY_HASH}'", "user_api_key_user_email": f"'{EMAIL}'"}
    }
    assert f"token='{FILTERED}'" in frame_vars["valid_token"]
    assert frame_vars["general_settings"] == {"master_key": FILTERED, "database_url": FILTERED}
    assert frame_vars["raw_headers"] == {"authorization": FILTERED, "x-api-key": FILTERED, "content-type": "'application/json'"}
    assert MASTER_KEY not in serialized
    assert VIRTUAL_KEY not in serialized
    assert "db-password-under-test" not in serialized


def test_transaction_events_are_scrubbed_too() -> None:
    transport: Final = RecordingTransport()
    client: Final = sentry_sdk.Client(transport=transport, **build_sentry_init_options(PII_OFF))
    client.capture_event(
        {
            "type": "transaction",
            "transaction": "/user/info",
            "contexts": {"trace": {"trace_id": "a" * 32, "span_id": "b" * 16}},
            "spans": [{"description": f"lookup {EMAIL} by {KEY_HASH}", "span_id": "c" * 16, "trace_id": "a" * 32}],
        }
    )
    assert transport.last_envelope is not None
    serialized: Final = json.dumps(transport.last_envelope.items[0].payload.json)
    assert EMAIL not in serialized
    assert KEY_HASH not in serialized
    assert f"lookup {FILTERED} by {FILTERED}" in serialized


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "UserAPIKeyAuth(token='abc', key_alias='team-a', user_id=None)",
            f"UserAPIKeyAuth(token='{FILTERED}', key_alias='team-a', user_id=None)",
        ),
        ('{"api_key": "sk-1", "model": "gpt-5"}', f'{{"api_key": "{FILTERED}", "model": "gpt-5"}}'),
        ("{'user_id': 'u-1', 'max_budget': 5}", f"{{'user_id': '{FILTERED}', 'max_budget': 5}}"),
        ("Config(OPENAI_API_KEY=sk-live, timeout=10)", f"Config(OPENAI_API_KEY='{FILTERED}', timeout=10)"),
        ("lookup for somebody@example.com failed", f"lookup for {FILTERED} failed"),
        (f"hashed key {KEY_HASH} not found", f"hashed key {FILTERED} not found"),
        ("request id 0123456789abcdef0123456789abcdef stays", "request id 0123456789abcdef0123456789abcdef stays"),
        ("monkey=banana", "monkey=banana"),
        (
            "{'x-api-key': 'k-1', 'cookie': 'session=abc', 'content-type': 'application/json'}",
            f"{{'x-api-key': '{FILTERED}', 'cookie': '{FILTERED}', 'content-type': 'application/json'}}",
        ),
        (
            "headers={'x-tenant-key': 'sk-custom-header-key-0123456789'} key_name='sk-...6789'",
            f"headers={{'x-tenant-key': '{FILTERED}'}} key_name='sk-...6789'",
        ),
        (
            "master_key={'value': 'not-a-litellm-key'} timeout=10",
            f"master_key='{FILTERED}' timeout=10",
        ),
        (
            "credentials=[{'value': ('deep', 'secret')}], model='gpt-5'",
            f"credentials='{FILTERED}', model='gpt-5'",
        ),
    ],
)
def test_string_scrubber_rewrites_field_and_value_forms(text: str, expected: str) -> None:
    assert build_string_scrubber(send_default_pii=False)(text) == expected


def test_json_walk_fails_closed_past_the_depth_cap() -> None:
    scrub: Final = build_string_scrubber(send_default_pii=False)
    nested: Final = reduce(lambda inner, _: [inner], range(MAX_SCRUB_DEPTH + 1), cast("JsonValue", "api_key=sk-1"))
    assert FILTERED in json.dumps(scrub_json_strings(nested, scrub))
    assert "sk-1" not in json.dumps(scrub_json_strings(nested, scrub))
    assert scrub_json_strings([["api_key=sk-1"]], scrub) == [[f"api_key='{FILTERED}'"]]


def test_string_scrubber_with_pii_on_only_scrubs_secrets() -> None:
    scrub: Final = build_string_scrubber(send_default_pii=True)
    assert scrub(f"user_id='{EMAIL}', token='{KEY_HASH}', email {EMAIL} hash {KEY_HASH}") == (
        f"user_id='{EMAIL}', token='{FILTERED}', email {EMAIL} hash {KEY_HASH}"
    )
    assert scrub(f"headers={{'authorization': 'Bearer {VIRTUAL_KEY}'}} sent {VIRTUAL_KEY}") == (
        f"headers={{'authorization': '{FILTERED}'}} sent {FILTERED}"
    )


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({}, False),
        ({"SENTRY_SEND_DEFAULT_PII": "true"}, True),
        ({"SENTRY_SEND_DEFAULT_PII": "True"}, True),
        ({"SENTRY_SEND_DEFAULT_PII": "false"}, False),
        ({"SENTRY_SEND_DEFAULT_PII": "yes please"}, False),
    ],
)
def test_send_default_pii_comes_from_the_environment(env: Mapping[str, str], expected: bool) -> None:
    assert build_sentry_init_options(env)["send_default_pii"] is expected


def test_init_options_read_dsn_rates_and_environment() -> None:
    options: Final = build_sentry_init_options(
        {
            "SENTRY_DSN": "https://key@sentry.example/7",
            "SENTRY_API_TRACE_RATE": "0.25",
            "SENTRY_API_SAMPLE_RATE": "0.5",
            "SENTRY_ENVIRONMENT": "staging",
        }
    )
    assert options["dsn"] == "https://key@sentry.example/7"
    assert options["traces_sample_rate"] == 0.25
    assert options["sample_rate"] == 0.5
    assert options["environment"] == "staging"
    assert options["event_scrubber"].recursive is True


def test_init_options_defaults() -> None:
    options: Final = build_sentry_init_options({})
    assert options["dsn"] is None
    assert options["traces_sample_rate"] == 1.0
    assert options["sample_rate"] == 1.0
    assert options["environment"] == "production"
