"""Tests for the hide-secrets guardrail (LIT-3548).

Covers the three defects from the ticket:
- ``apply_guardrail`` (the UI test playground path) must redact, not echo.
- Guardrail runs must record ``standard_logging_guardrail_information`` so
  Spend Logs / the guardrails monitor show activity, with hits ("mask" +
  masked_entity_count) distinguishable from clean requests ("allow").
- Defining ``apply_guardrail`` must NOT reroute proxied traffic off the
  native ``async_pre_call_hook`` (per-key opt-out and ``data["prompt"]``
  handling live only on the native path).
"""

import time

import pytest

from litellm_enterprise.enterprise_callbacks.secret_detection import (
    _ENTERPRISE_SecretDetection,
)
from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth

AWS_KEY = "AKIAIOSFODNN7EXAMPLE"
OPENAI_KEY = "sk-test-abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGH"
SHORT_OPENAI_KEY = "sk-12345"
UNICODE_DIGIT_SUFFIX = "sk-notification٣"
STRIPE_LIVE_KEY = f"sk_live_{'1234567890' * 3}"
URL_ENCODED_KEY = "Bearer%20sk-Ab3dEf6Gh7Ij8Kl9Mn0Pq2Rs3Tu4Vw5X"
AWS_KEYS = [f"AKIAIOSFODNN7EXAMPL{suffix}" for suffix in "FEDCBA"]


def _guardrail() -> _ENTERPRISE_SecretDetection:
    return _ENTERPRISE_SecretDetection(guardrail_name="hide-secrets", event_hook="pre_call", default_on=True)


def _recorded(request_data: dict) -> dict:
    entries = request_data["metadata"]["standard_logging_guardrail_information"]
    assert len(entries) == 1
    return entries[0]


def test_scan_message_preserves_benign_identifiers_and_xml_tags():
    guardrail = _guardrail()
    content = "<task-notification> model: claude-sonnet-4-5-20250929 </task-notification>"

    assert guardrail.scan_message_for_secrets(content) == []
    assert guardrail.redact_text(content) == content
    assert guardrail.redact_text("result = compute(x) </task-notification>") == (
        "result = compute(x) </task-notification>"
    )


def test_scan_message_preserves_quoted_benign_identifiers():
    guardrail = _guardrail()
    content = '{"content-type": "application/json", "model": "claude-sonnet-4-5-20250929"}'

    assert guardrail.scan_message_for_secrets(content) == []
    assert guardrail.redact_text(content) == content


def test_scan_message_redacts_every_openai_key_occurrence():
    guardrail = _guardrail()
    content = f"first {OPENAI_KEY}, second {OPENAI_KEY}"

    assert guardrail.redact_text(content) == "first [REDACTED], second [REDACTED]"


def test_scan_message_redacts_short_numeric_openai_like_values():
    guardrail = _guardrail()

    assert guardrail.redact_text(f"value {SHORT_OPENAI_KEY}") == "value [REDACTED]"


def test_scan_message_requires_ascii_digits_for_openai_like_values():
    guardrail = _guardrail()

    assert guardrail.scan_message_for_secrets(UNICODE_DIGIT_SUFFIX) == []
    assert guardrail.redact_text(UNICODE_DIGIT_SUFFIX) == UNICODE_DIGIT_SUFFIX


def test_scan_message_redacts_openai_key_after_separator():
    guardrail = _guardrail()

    assert guardrail.redact_text(f"openai_{OPENAI_KEY} key-{OPENAI_KEY}") == (
        "openai_[REDACTED] key-[REDACTED]"
    )
    assert guardrail.redact_text(URL_ENCODED_KEY) == "Bearer%20[REDACTED]"


def test_scan_message_does_not_stop_openai_key_at_token_characters():
    guardrail = _guardrail()

    assert guardrail.redact_text("key sk-proj-abcde12345/extra") == "key [REDACTED]/extra"


def test_scan_message_stays_linear_on_repeated_sk_separators():
    guardrail = _guardrail()
    content = "-sk-" * 25_000

    started = time.perf_counter()
    assert guardrail.scan_message_for_secrets(content) == []
    assert time.perf_counter() - started < 2.0


def test_scan_message_redacts_whole_stripe_live_key():
    guardrail = _guardrail()

    assert guardrail.redact_text(f"stripe {STRIPE_LIVE_KEY} end") == "stripe [REDACTED] end"


def test_scan_message_returns_matches_in_stable_order():
    guardrail = _guardrail()
    detected = guardrail.scan_message_for_secrets(" ".join(AWS_KEYS))

    assert [secret["value"] for secret in detected] == sorted(AWS_KEYS)


def test_scan_message_replaces_longest_overlapping_match_first():
    guardrail = _guardrail()
    content = f'token = "{OPENAI_KEY}/extra"'

    detected = guardrail.scan_message_for_secrets(content)
    assert [secret["value"] for secret in detected] == [f"{OPENAI_KEY}/extra", OPENAI_KEY]
    assert guardrail.redact_text(content) == 'token = "[REDACTED]"'


@pytest.mark.asyncio
async def test_apply_guardrail_redacts_secrets():
    """Playground path: the returned texts must carry [REDACTED], not the secret."""
    guardrail = _guardrail()
    request_data: dict = {"metadata": {}}

    result = await guardrail.apply_guardrail(
        inputs={"texts": [f"my key is {AWS_KEY}, keep it safe"]},
        request_data=request_data,
        input_type="request",
    )

    assert result["texts"] == ["my key is [REDACTED], keep it safe"]

    recorded = _recorded(request_data)
    assert recorded["guardrail_status"] == "success"
    assert recorded["guardrail_response"] == "mask"
    assert recorded["guardrail_provider"] == "hide-secrets"
    assert recorded["masked_entity_count"] == {"AWS Access Key": 1}


@pytest.mark.asyncio
async def test_apply_guardrail_clean_text_records_allow():
    guardrail = _guardrail()
    request_data: dict = {"metadata": {}}

    result = await guardrail.apply_guardrail(
        inputs={"texts": ["nothing sensitive here"]},
        request_data=request_data,
        input_type="request",
    )

    assert result["texts"] == ["nothing sensitive here"]

    recorded = _recorded(request_data)
    assert recorded["guardrail_status"] == "success"
    assert recorded["guardrail_response"] == "allow"
    assert recorded["masked_entity_count"] == {}


@pytest.mark.asyncio
async def test_pre_call_hook_records_mask_with_entity_count():
    """Live-traffic path: a redaction must be visible in spend-log telemetry."""
    guardrail = _guardrail()
    data = {
        "messages": [{"role": "user", "content": f"use {AWS_KEY} for auth"}],
        "metadata": {},
    }

    await guardrail.async_pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(),
        cache=DualCache(),
        data=data,
        call_type="completion",
    )

    assert data["messages"][0]["content"] == "use [REDACTED] for auth"

    recorded = _recorded(data)
    assert recorded["guardrail_status"] == "success"
    assert recorded["guardrail_response"] == "mask"
    assert recorded["guardrail_provider"] == "hide-secrets"
    assert recorded["masked_entity_count"] == {"AWS Access Key": 1}


@pytest.mark.asyncio
async def test_pre_call_hook_clean_request_records_allow():
    """A request with no secrets must be distinguishable from a redacted one."""
    guardrail = _guardrail()
    data = {
        "messages": [{"role": "user", "content": "what's the weather"}],
        "metadata": {},
    }

    await guardrail.async_pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(),
        cache=DualCache(),
        data=data,
        call_type="completion",
    )

    recorded = _recorded(data)
    assert recorded["guardrail_status"] == "success"
    assert recorded["guardrail_response"] == "allow"
    assert recorded["masked_entity_count"] == {}


@pytest.mark.asyncio
async def test_pre_call_hook_opt_out_records_nothing():
    """A key with permissions={"hide_secrets": False} skips redaction, so no
    telemetry is recorded: every reader of a recorded entry (guardrail usage
    tracking, compliance checks, the spend-log viewer) counts it as a run."""
    guardrail = _guardrail()
    content = f"my key is {AWS_KEY}"
    data = {"messages": [{"role": "user", "content": content}], "metadata": {}}

    await guardrail.async_pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(permissions={"hide_secrets": False}),
        cache=DualCache(),
        data=data,
        call_type="completion",
    )

    assert data["messages"][0]["content"] == content  # untouched
    assert "standard_logging_guardrail_information" not in data["metadata"]


@pytest.mark.asyncio
async def test_pre_call_hook_still_redacts_text_completion_prompt():
    """data["prompt"] (str and list) is a native-hook-only surface; it must
    keep redacting now that the class also implements apply_guardrail."""
    guardrail = _guardrail()
    data = {"prompt": f"key {AWS_KEY} end", "metadata": {}}
    await guardrail.async_pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(),
        cache=DualCache(),
        data=data,
        call_type="completion",
    )
    assert data["prompt"] == "key [REDACTED] end"

    guardrail = _guardrail()
    data = {"prompt": [f"key {AWS_KEY}", "clean"], "metadata": {}}
    await guardrail.async_pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(),
        cache=DualCache(),
        data=data,
        call_type="completion",
    )
    assert data["prompt"] == ["key [REDACTED]", "clean"]


def test_proxied_traffic_stays_on_native_hooks():
    """Implementing apply_guardrail must not reroute proxied requests onto the
    unified path: that path skips ``should_run_check`` (per-key opt-out) and
    never sees ``data["prompt"]``."""
    guardrail = _guardrail()
    assert guardrail.uses_apply_guardrail_interface() is True
    assert guardrail._deployment_pre_call_target() is guardrail


@pytest.mark.asyncio
async def test_apply_guardrail_without_texts_records_nothing():
    """No inputs means nothing was inspected, so no "allow" row is recorded.
    Empty strings count as no input: there is no content to inspect."""
    guardrail = _guardrail()

    empty_variants: list[list[str]] = [[], ["", ""]]
    for texts in empty_variants:
        request_data: dict = {"metadata": {}}
        result = await guardrail.apply_guardrail(
            inputs={"texts": texts}, request_data=request_data, input_type="request"
        )
        assert result == {"texts": texts}
        assert "standard_logging_guardrail_information" not in request_data["metadata"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "data",
    [
        pytest.param(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {"url": "https://x/y.png"}}],
                    }
                ],
                "metadata": {},
            },
            id="image_only",
        ),
        pytest.param(
            {"messages": [{"role": "user", "content": ""}], "metadata": {}},
            id="empty_message",
        ),
        pytest.param({"prompt": "", "metadata": {}}, id="empty_prompt"),
        pytest.param({"prompt": ["", ""], "metadata": {}}, id="empty_prompt_list"),
    ],
)
async def test_pre_call_hook_without_inspectable_text_records_nothing(data: dict):
    """A payload the guardrail could not inspect (image-only content, empty
    strings) must not record an "allow" run: monitoring would count a check
    that never looked at any text."""
    guardrail = _guardrail()

    await guardrail.async_pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(),
        cache=DualCache(),
        data=data,
        call_type="completion",
    )

    assert "standard_logging_guardrail_information" not in data["metadata"]


@pytest.mark.asyncio
async def test_pre_call_hook_mixed_prompt_list_still_redacts_and_records():
    """A prompt list mixing empty and real strings is inspected, so the run is
    recorded and the non-empty entry is still redacted."""
    guardrail = _guardrail()
    data = {"prompt": ["", f"key {AWS_KEY}"], "metadata": {}}

    await guardrail.async_pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(),
        cache=DualCache(),
        data=data,
        call_type="completion",
    )

    assert data["prompt"] == ["", "key [REDACTED]"]
    recorded = _recorded(data)
    assert recorded["guardrail_response"] == "mask"
    assert recorded["masked_entity_count"] == {"AWS Access Key": 1}


@pytest.mark.asyncio
async def test_legacy_nameless_instance_records_nothing():
    """``litellm_settings.callbacks: ["hide_secrets"]`` builds an arg-less
    instance with no guardrail_name. It still redacts, but recording a nameless
    entry would flip every spend row's guardrail status with nothing to join on."""
    guardrail = _ENTERPRISE_SecretDetection()
    data = {
        "messages": [{"role": "user", "content": f"use {AWS_KEY} for auth"}],
        "metadata": {},
    }

    await guardrail.async_pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(),
        cache=DualCache(),
        data=data,
        call_type="completion",
    )

    assert data["messages"][0]["content"] == "use [REDACTED] for auth"
    assert "standard_logging_guardrail_information" not in data["metadata"]
