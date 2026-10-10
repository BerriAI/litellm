import asyncio
import json

import pytest


import litellm
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig
from litellm.llms.bedrock.chat.invoke_transformations.amazon_openai_transformation import (
    AmazonBedrockOpenAIConfig,
)
from litellm.llms.bedrock.chat.invoke_transformations.base_invoke_transformation import (
    AmazonInvokeConfig,
)
from litellm.llms.bedrock.messages.invoke_transformations.anthropic_claude3_transformation import (
    AmazonAnthropicClaudeMessagesConfig,
)
from litellm.llms.bedrock.request_metadata import (
    BEDROCK_REQUEST_METADATA_HEADER,
    BEDROCK_REQUEST_METADATA_MAX_PAIRS,
    resolve_bedrock_request_metadata,
)

MODEL = "anthropic.claude-3-5-sonnet-20240620-v1:0"
MESSAGES = [{"role": "user", "content": "hi"}]
ALL_FIELDS = [
    "user_api_key_alias",
    "user_api_key_team_alias",
    "user_api_key_user_email",
    "spend_logs_metadata",
]
IDENTITY = {"user_api_key_alias": "prod-key", "user_api_key_team_alias": "platform"}


def litellm_params(metadata_key, **metadata):
    return {metadata_key: dict(metadata)}


def converse_body(litellm_params_value, optional_params=None):
    return AmazonConverseConfig()._transform_request(
        model=MODEL,
        messages=MESSAGES,
        optional_params=dict(optional_params or {}),
        litellm_params=dict(litellm_params_value),
    )


def converse_body_async(litellm_params_value, optional_params=None):
    """The proxy serves completions through the async transform, so every rule asserted against
    the sync body has to be asserted against this one too or half the product is untested."""
    return asyncio.run(
        AmazonConverseConfig()._async_transform_request(
            model=MODEL,
            messages=MESSAGES,
            optional_params=dict(optional_params or {}),
            litellm_params=dict(litellm_params_value),
        )
    )


CONVERSE_DRIVERS = [converse_body, converse_body_async]


@pytest.mark.parametrize("setting", [None, []])
def test_feature_off_by_default_leaves_body_and_headers_untouched(setting, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "bedrock_request_metadata_fields", setting)
    params = litellm_params("metadata", spend_logs_metadata={"team": "x"}, **IDENTITY)

    assert "requestMetadata" not in converse_body(params)
    assert BEDROCK_REQUEST_METADATA_HEADER not in AmazonInvokeConfig().validate_environment(
        headers={}, model=MODEL, messages=MESSAGES, optional_params={}, litellm_params=dict(params)
    )
    messages_headers, _ = AmazonAnthropicClaudeMessagesConfig().validate_anthropic_messages_environment(
        headers={}, model=MODEL, messages=MESSAGES, optional_params={}, litellm_params=dict(params)
    )
    assert BEDROCK_REQUEST_METADATA_HEADER not in messages_headers


@pytest.mark.parametrize("metadata_key", ["metadata", "litellm_metadata"])
def test_resolver_reads_both_metadata_variable_names(metadata_key, monkeypatch: pytest.MonkeyPatch):
    """`/v1/chat/completions` populates `metadata`; the LITELLM_METADATA_ROUTES populate
    `litellm_metadata`. Reading only one silently forwards nothing on the other route."""
    monkeypatch.setattr(litellm, "bedrock_request_metadata_fields", ALL_FIELDS)
    params = litellm_params(metadata_key, spend_logs_metadata={"cost_center": "cc-1"}, **IDENTITY)

    assert converse_body(params)["requestMetadata"] == {**IDENTITY, "cost_center": "cc-1"}


@pytest.mark.parametrize("metadata_key", ["metadata", "litellm_metadata"])
def test_invoke_messages_header_reads_both_metadata_variable_names(metadata_key, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "bedrock_request_metadata_fields", ALL_FIELDS)
    params = litellm_params(metadata_key, **IDENTITY)

    headers, _ = AmazonAnthropicClaudeMessagesConfig().validate_anthropic_messages_environment(
        headers={}, model=MODEL, messages=MESSAGES, optional_params={}, litellm_params=params
    )

    assert json.loads(headers[BEDROCK_REQUEST_METADATA_HEADER]) == IDENTITY


@pytest.mark.parametrize("reverse_client_keys", [False, True])
@pytest.mark.parametrize("field_order", [ALL_FIELDS, list(reversed(ALL_FIELDS))])
@pytest.mark.parametrize("client_source", ["spend_logs_metadata", "requestMetadata"])
def test_identity_survives_a_caller_filling_every_slot(
    reverse_client_keys,
    field_order,
    client_source,
    monkeypatch: pytest.MonkeyPatch,
):
    """A caller sending 16 keys of its own must not evict the identity the feature exists to
    produce. Driven over every input ordering so the invariant is not an accident of one."""
    monkeypatch.setattr(litellm, "bedrock_request_metadata_fields", field_order)
    client_keys = [f"client_{index:02d}" for index in range(BEDROCK_REQUEST_METADATA_MAX_PAIRS)]
    client_pairs = {key: "v" for key in (reversed(client_keys) if reverse_client_keys else client_keys)}
    if client_source == "spend_logs_metadata":
        params, optional_params = litellm_params("metadata", spend_logs_metadata=client_pairs, **IDENTITY), {}
    else:
        params, optional_params = litellm_params("metadata", **IDENTITY), {"requestMetadata": client_pairs}

    resolved = converse_body(params, optional_params)["requestMetadata"]

    assert len(resolved) == BEDROCK_REQUEST_METADATA_MAX_PAIRS
    for key, value in IDENTITY.items():
        assert resolved[key] == value
    assert len([key for key in resolved if key.startswith("client_")]) == (
        BEDROCK_REQUEST_METADATA_MAX_PAIRS - len(IDENTITY)
    )


@pytest.mark.parametrize(
    "field_order",
    [
        ["user_api_key_alias", "user_api_key_alias", "user_api_key_team_alias", "spend_logs_metadata"],
        ["user_api_key_alias", "user_api_key_team_alias", "user_api_key_alias", "spend_logs_metadata"],
        ["user_api_key_alias", "user_api_key_team_alias", "spend_logs_metadata", "user_api_key_team_alias"],
    ],
)
def test_a_field_repeated_in_the_allow_list_does_not_consume_a_client_slot(
    field_order,
    monkeypatch: pytest.MonkeyPatch,
):
    """An operator repeating a field in YAML must not inflate the reserved count and shrink the
    client budget. Asserts the client keys that should have fitted actually reach the wire, since
    asserting only that identity survives passes with or without the deduplication."""
    monkeypatch.setattr(litellm, "bedrock_request_metadata_fields", field_order)
    client_keys = [f"client_{index:02d}" for index in range(BEDROCK_REQUEST_METADATA_MAX_PAIRS - 1)]
    params = litellm_params("metadata", spend_logs_metadata={key: "v" for key in client_keys}, **IDENTITY)

    resolved = converse_body(params)["requestMetadata"]

    expected_client_slots = BEDROCK_REQUEST_METADATA_MAX_PAIRS - len(IDENTITY)
    assert resolved == {**IDENTITY, **{key: "v" for key in client_keys[:expected_client_slots]}}
    assert len(resolved) == BEDROCK_REQUEST_METADATA_MAX_PAIRS
    assert client_keys[expected_client_slots - 1] in resolved


@pytest.mark.parametrize("client_source", ["spend_logs_metadata", "requestMetadata"])
@pytest.mark.parametrize(
    "forged_key",
    ["user_api_key_team_alias", "user_api_key_org_alias", "user_api_key_hash"],
)
def test_caller_cannot_forge_or_shadow_a_reserved_identity_key(
    forged_key,
    client_source,
    monkeypatch: pytest.MonkeyPatch,
):
    """`user_api_key_org_alias` and `user_api_key_hash` are names the proxy does not set here,
    so an exact-key reservation would let the forged value through under a name that reads as
    proxy-authoritative in the AWS billing record."""
    monkeypatch.setattr(litellm, "bedrock_request_metadata_fields", ALL_FIELDS)
    forged = {forged_key: "attacker-controlled"}
    if client_source == "spend_logs_metadata":
        params, optional_params = litellm_params("metadata", spend_logs_metadata=forged, **IDENTITY), {}
    else:
        params, optional_params = litellm_params("metadata", **IDENTITY), {"requestMetadata": forged}

    resolved = converse_body(params, optional_params)["requestMetadata"]

    assert resolved == IDENTITY
    assert "attacker-controlled" not in resolved.values()


def test_identity_violating_the_character_class_is_dropped_and_the_request_succeeds(monkeypatch: pytest.MonkeyPatch):
    """A team alias with an apostrophe must not turn a working request into a 400 the moment
    an operator flips the setting on."""
    monkeypatch.setattr(litellm, "bedrock_request_metadata_fields", ALL_FIELDS)
    params = litellm_params(
        "metadata",
        user_api_key_alias="prod-key",
        user_api_key_team_alias="O'Brien's team",
        user_api_key_user_email="x" * 300,
    )

    body = converse_body(params)

    assert body["requestMetadata"] == {"user_api_key_alias": "prod-key"}
    assert body["messages"]


def test_caller_supplied_violation_still_raises_bad_request(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "bedrock_request_metadata_fields", ALL_FIELDS)

    with pytest.raises(litellm.exceptions.BadRequestError):
        converse_body(
            litellm_params("metadata", **IDENTITY),
            {"requestMetadata": {"team": "O'Brien's team"}},
        )


def test_non_string_and_absent_identity_values_are_dropped(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "bedrock_request_metadata_fields", ALL_FIELDS + ["user_api_key_spend"])
    params = litellm_params("metadata", user_api_key_alias="prod-key", user_api_key_spend=1.25)

    assert converse_body(params)["requestMetadata"] == {"user_api_key_alias": "prod-key"}


def test_email_is_separately_opt_in(monkeypatch: pytest.MonkeyPatch):
    """PII crossing into CloudTrail only when the operator names the field."""
    identity_with_email = {**IDENTITY, "user_api_key_user_email": "owner@example.com"}
    monkeypatch.setattr(litellm, "bedrock_request_metadata_fields", ["user_api_key_alias", "user_api_key_team_alias"])
    assert (
        "user_api_key_user_email"
        not in converse_body(litellm_params("metadata", **identity_with_email))["requestMetadata"]
    )

    monkeypatch.setattr(litellm, "bedrock_request_metadata_fields", ALL_FIELDS)
    assert converse_body(litellm_params("metadata", **identity_with_email))["requestMetadata"] == identity_with_email


def test_resolver_returns_none_when_nothing_survives(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "bedrock_request_metadata_fields", ALL_FIELDS)
    assert resolve_bedrock_request_metadata(litellm_params=None) is None
    assert resolve_bedrock_request_metadata(litellm_params={"metadata": {"unrelated": "x"}}) is None


def test_invoke_header_is_json_encoded_and_signed(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "bedrock_request_metadata_fields", ALL_FIELDS)
    params = litellm_params("metadata", spend_logs_metadata={"cost_center": "cc-1"}, **IDENTITY)

    headers = AmazonInvokeConfig().validate_environment(
        headers={"anthropic-version": "bedrock-2023-05-31"},
        model=MODEL,
        messages=MESSAGES,
        optional_params={},
        litellm_params=params,
    )

    assert json.loads(headers[BEDROCK_REQUEST_METADATA_HEADER]) == {**IDENTITY, "cost_center": "cc-1"}
    signed = BaseAWSLLM()._filter_headers_for_aws_signature(headers)
    assert BEDROCK_REQUEST_METADATA_HEADER in signed
    assert "anthropic-version" not in signed


def test_a_caller_supplied_guardrail_header_still_wins(monkeypatch: pytest.MonkeyPatch):
    """The no-displace rule is deliberate for the guardrail headers and must survive the
    request-metadata header becoming proxy-owned."""
    monkeypatch.setattr(litellm, "bedrock_request_metadata_fields", ALL_FIELDS)

    headers = AmazonInvokeConfig().validate_environment(
        headers={"X-Amzn-Bedrock-GuardrailIdentifier": "caller-set"},
        model=MODEL,
        messages=MESSAGES,
        optional_params={"guardrailConfig": {"guardrailIdentifier": "gid", "guardrailVersion": "DRAFT"}},
        litellm_params=litellm_params("metadata", **IDENTITY),
    )

    assert headers["X-Amzn-Bedrock-GuardrailIdentifier"] == "caller-set"
    assert headers["X-Amzn-Bedrock-GuardrailVersion"] == "DRAFT"


FORGED = '{"user_api_key_alias":"FORGED-KEY","user_api_key_team_alias":"FORGED-TEAM"}'


def invoke_headers(caller_headers, params, optional_params=None):
    return AmazonInvokeConfig().validate_environment(
        headers=dict(caller_headers),
        model=MODEL,
        messages=MESSAGES,
        optional_params=dict(optional_params or {}),
        litellm_params=dict(params),
    )


def messages_headers(caller_headers, params):
    resolved, _ = AmazonAnthropicClaudeMessagesConfig().validate_anthropic_messages_environment(
        headers=dict(caller_headers),
        model=MODEL,
        messages=MESSAGES,
        optional_params={},
        litellm_params=dict(params),
    )
    return resolved


def openai_invoke_headers(caller_headers, params):
    return AmazonBedrockOpenAIConfig().validate_environment(
        headers=dict(caller_headers),
        model=MODEL,
        messages=MESSAGES,
        optional_params={},
        litellm_params=dict(params),
    )


def converse_headers(caller_headers, params):
    return AmazonConverseConfig().validate_environment(
        headers=dict(caller_headers),
        model=MODEL,
        messages=MESSAGES,
        optional_params={},
        litellm_params=dict(params),
    )


HEADER_DRIVERS = [invoke_headers, messages_headers, openai_invoke_headers, converse_headers]


def metadata_header_values(headers):
    return [value for name, value in headers.items() if name.lower() == BEDROCK_REQUEST_METADATA_HEADER.lower()]


def test_converse_still_sets_the_bearer_authorization_header(monkeypatch: pytest.MonkeyPatch):
    """Converse owns the metadata header now, and that must not disturb the api_key path its
    validate_environment existed for. Closing the forgery hole cannot break authentication."""
    monkeypatch.setattr(litellm, "bedrock_request_metadata_fields", ALL_FIELDS)

    headers = AmazonConverseConfig().validate_environment(
        headers={},
        model=MODEL,
        messages=MESSAGES,
        optional_params={},
        litellm_params=dict(litellm_params("metadata", **IDENTITY)),
        api_key="sk-converse-bearer",
    )

    assert headers["Authorization"] == "Bearer sk-converse-bearer"
    assert metadata_header_values(headers) == [json.dumps(IDENTITY, separators=(",", ":"))]


@pytest.mark.parametrize("driver", HEADER_DRIVERS)
@pytest.mark.parametrize(
    "caller_header_name",
    [BEDROCK_REQUEST_METADATA_HEADER, BEDROCK_REQUEST_METADATA_HEADER.lower(), "x-AMZN-bedrock-Request-METADATA"],
)
def test_a_caller_cannot_forge_the_request_metadata_header(driver, caller_header_name, monkeypatch: pytest.MonkeyPatch):
    """`extra_headers` puts caller-supplied names into the same dict the proxy merges into, so a
    deferring merge would sign the caller's forged identity into the AWS billing record. Every
    spelling must lose, or a second variant is left for the transport to choose between."""
    monkeypatch.setattr(litellm, "bedrock_request_metadata_fields", ALL_FIELDS)

    headers = driver({caller_header_name: FORGED}, litellm_params("metadata", **IDENTITY))

    values = metadata_header_values(headers)
    assert values == [json.dumps(IDENTITY, separators=(",", ":"))]
    assert "FORGED" not in json.dumps(headers)


@pytest.mark.parametrize("driver", HEADER_DRIVERS)
def test_a_caller_cannot_forge_the_header_when_the_resolver_yields_nothing(driver, monkeypatch: pytest.MonkeyPatch):
    """Forwarding enabled but nothing resolvable, which a caller can arrange by supplying values
    that all fail Bedrock's rules. Owned-but-empty must mean no header on the wire, never a
    fallback to the caller's."""
    monkeypatch.setattr(litellm, "bedrock_request_metadata_fields", ALL_FIELDS)
    unresolvable = litellm_params("metadata", user_api_key_alias="O'Brien's key", user_api_key_team_alias="x" * 300)

    headers = driver({BEDROCK_REQUEST_METADATA_HEADER: FORGED}, unresolvable)

    assert metadata_header_values(headers) == []
    assert "FORGED" not in json.dumps(headers)


@pytest.mark.parametrize("driver", CONVERSE_DRIVERS)
@pytest.mark.parametrize(
    "forged_key",
    ["user_api_key_team_alias", "user_api_key_org_alias", "user_api_key_hash"],
)
def test_a_caller_cannot_keep_reserved_body_keys_when_the_resolver_yields_nothing(
    forged_key,
    driver,
    monkeypatch: pytest.MonkeyPatch,
):
    """The Converse body has the same fail-open shape as the header: with forwarding on and
    nothing resolvable, leaving the caller's `requestMetadata` in place would keep their
    reserved-prefix keys on the wire. Owned-but-empty must remove the field outright."""
    monkeypatch.setattr(litellm, "bedrock_request_metadata_fields", ALL_FIELDS)

    body = driver(litellm_params("metadata"), {"requestMetadata": {forged_key: "FORGED"}})

    assert "requestMetadata" not in body
    assert "FORGED" not in json.dumps(body)


@pytest.mark.parametrize("driver", CONVERSE_DRIVERS)
def test_benign_caller_body_metadata_still_survives_when_no_identity_resolves(driver, monkeypatch: pytest.MonkeyPatch):
    """Removing the field must be scoped to the reserved keys being the only thing left, not a
    blanket drop of the caller's own attribution pairs."""
    monkeypatch.setattr(litellm, "bedrock_request_metadata_fields", ALL_FIELDS)

    body = driver(
        litellm_params("metadata"),
        {"requestMetadata": {"cost_center": "cc-9", "user_api_key_team_alias": "FORGED"}},
    )

    assert body["requestMetadata"] == {"cost_center": "cc-9"}


@pytest.mark.parametrize("driver", CONVERSE_DRIVERS)
def test_caller_body_metadata_is_left_alone_when_forwarding_is_off(driver, monkeypatch: pytest.MonkeyPatch):
    """With the feature off the proxy does not own the field, so the pre-existing pass-through
    behaviour for a caller-supplied `requestMetadata` must be unchanged."""
    monkeypatch.setattr(litellm, "bedrock_request_metadata_fields", None)
    caller_supplied = {"user_api_key_team_alias": "caller-set", "cost_center": "cc-9"}

    body = driver(litellm_params("metadata", **IDENTITY), {"requestMetadata": caller_supplied})

    assert body["requestMetadata"] == caller_supplied


@pytest.mark.parametrize("driver", HEADER_DRIVERS)
def test_a_caller_header_is_left_alone_when_forwarding_is_off(driver, monkeypatch: pytest.MonkeyPatch):
    """The proxy only claims the name when the operator turned forwarding on; with the feature
    off this is an ordinary passthrough header and stripping it would be a regression."""
    monkeypatch.setattr(litellm, "bedrock_request_metadata_fields", None)

    headers = driver({BEDROCK_REQUEST_METADATA_HEADER: FORGED}, litellm_params("metadata", **IDENTITY))

    assert metadata_header_values(headers) == [FORGED]
