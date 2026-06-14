import copy
import sys
import os
from types import SimpleNamespace

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.proxy.common_utils.callback_utils import (
    add_policy_to_applied_policies_header,
    decrypt_callback_vars,
    encrypt_callback_vars,
    get_logging_caching_headers,
    initialize_callbacks_on_proxy,
    get_remaining_tokens_and_requests_from_request_data,
    normalize_callback_names,
    sanitize_openai_provider_metadata,
)
import litellm

from unittest.mock import patch
from litellm.proxy.common_utils.callback_utils import process_callback


def test_get_remaining_tokens_and_requests_from_request_data():
    model_group = "openrouter/google/gemini-2.0-flash-001"
    casedata = {
        "metadata": {
            "model_group": model_group,
            f"litellm-key-remaining-requests-{model_group}": 100,
            f"litellm-key-remaining-tokens-{model_group}": 200,
        }
    }

    headers = get_remaining_tokens_and_requests_from_request_data(casedata)

    expected_name = "openrouter-google-gemini-2.0-flash-001"
    assert headers == {
        f"x-litellm-key-remaining-requests-{expected_name}": 100,
        f"x-litellm-key-remaining-tokens-{expected_name}": 200,
    }


@patch(
    "litellm.proxy.common_utils.callback_utils.CustomLogger.get_callback_env_vars",
    return_value=["API_KEY", "MISSING_VAR"],
)
def test_process_callback_with_env_vars(mock_get_env_vars):
    environment_variables = {
        "API_KEY": "PLAIN_VALUE",
        "UNUSED": "SHOULD_BE_IGNORED",
    }

    result = process_callback(
        _callback="my_callback",
        callback_type="input",
        environment_variables=environment_variables,
    )

    assert result["name"] == "my_callback"
    assert result["type"] == "input"
    assert result["variables"] == {
        "API_KEY": "PLAIN_VALUE",
        "MISSING_VAR": None,
    }


@patch(
    "litellm.proxy.common_utils.callback_utils.CustomLogger.get_callback_env_vars",
    return_value=[],
)
def test_process_callback_with_no_required_env_vars(mock_get_env_vars):
    result = process_callback(
        _callback="another_callback",
        callback_type="output",
        environment_variables={"SHOULD_NOT_BE_USED": "VALUE"},
    )

    assert result["name"] == "another_callback"
    assert result["type"] == "output"
    assert result["variables"] == {}


def test_normalize_callback_names_none_returns_empty_list():
    assert normalize_callback_names(None) == []
    assert normalize_callback_names([]) == []


def test_normalize_callback_names_lowercases_strings():
    assert normalize_callback_names(["SQS", "S3", "CUSTOM_CALLBACK"]) == [
        "sqs",
        "s3",
        "custom_callback",
    ]


def test_add_policy_to_applied_policies_header_uses_litellm_metadata_bucket():
    request_data = {
        "input_file_id": "file-abc123",
        "litellm_metadata": {},
    }

    add_policy_to_applied_policies_header(
        request_data=request_data, policy_name="global-baseline"
    )

    assert request_data["litellm_metadata"]["applied_policies"] == ["global-baseline"]
    assert "applied_policies" not in request_data.get("metadata", {})


def test_sanitize_openai_provider_metadata_strips_internal_tracking_fields():
    metadata = {
        "customer_id": "cust-123",
        "applied_policies": ["global-baseline"],
        "applied_guardrails": ["pii_blocker"],
        "note": 42,
    }

    sanitized = sanitize_openai_provider_metadata(metadata)

    assert sanitized == {"customer_id": "cust-123"}


def test_get_logging_caching_headers_merges_metadata_and_litellm_metadata():
    request_data = {
        "metadata": {"customer_id": "cust-123"},
        "litellm_metadata": {
            "applied_policies": ["global-baseline"],
            "applied_guardrails": ["pii_blocker"],
            "policy_sources": {"global-baseline": "team_default"},
        },
    }

    headers = get_logging_caching_headers(request_data)

    assert headers["x-litellm-applied-policies"] == "global-baseline"
    assert headers["x-litellm-applied-guardrails"] == "pii_blocker"
    assert headers["x-litellm-policy-sources"] == "global-baseline=team_default"


def test_initialize_callbacks_on_proxy_instantiates_compression_interception(
    monkeypatch,
):
    dummy_callback = object()
    monkeypatch.setitem(
        sys.modules,
        "litellm.proxy.proxy_server",
        SimpleNamespace(prisma_client=None),
    )
    monkeypatch.setattr(
        "litellm.integrations.compression_interception.handler.CompressionInterceptionLogger.initialize_from_proxy_config",
        lambda litellm_settings, callback_specific_params: dummy_callback,
    )

    original_callbacks = (
        list(litellm.callbacks) if isinstance(litellm.callbacks, list) else []
    )
    litellm.callbacks = []
    try:
        initialize_callbacks_on_proxy(
            value=["compression_interception"],
            premium_user=False,
            config_file_path=".",
            litellm_settings={"compression_interception_params": {"enabled": True}},
            callback_specific_params={},
        )
        assert dummy_callback in litellm.callbacks
        assert "compression_interception" not in litellm.callbacks
    finally:
        litellm.callbacks = original_callbacks


# ---------------------------------------------------------------------------
# encrypt_callback_vars / decrypt_callback_vars
# ---------------------------------------------------------------------------


def _sample_metadata():
    return {
        "logging": [
            {
                "callback_name": "langfuse",
                "callback_type": "success_and_failure",
                "callback_vars": {
                    "langfuse_public_key": "pk-lf-public",
                    "langfuse_secret_key": "sk-lf-secret",
                    "langfuse_host": "https://cloud.langfuse.com",
                },
            }
        ],
        "callback_settings": {
            "callback_vars": {"langsmith_api_key": "ls-api-key"},
        },
        "tags": ["unrelated"],
    }


def _set_salt_key(monkeypatch):
    monkeypatch.setenv("LITELLM_SALT_KEY", "test-salt-32-bytes-aaaaaaaaaaaaaa")


def test_encrypt_callback_vars_round_trip(monkeypatch):
    _set_salt_key(monkeypatch)
    original = _sample_metadata()
    encrypted = encrypt_callback_vars(original)

    enc_vars = encrypted["logging"][0]["callback_vars"]
    assert enc_vars["langfuse_secret_key"] != "sk-lf-secret"
    assert enc_vars["langfuse_public_key"] != "pk-lf-public"
    assert (
        encrypted["callback_settings"]["callback_vars"]["langsmith_api_key"]
        != "ls-api-key"
    )

    decrypted = decrypt_callback_vars(encrypted)
    assert (
        decrypted["logging"][0]["callback_vars"]
        == original["logging"][0]["callback_vars"]
    )
    assert (
        decrypted["callback_settings"]["callback_vars"]
        == original["callback_settings"]["callback_vars"]
    )


def test_encrypt_callback_vars_is_idempotent(monkeypatch):
    _set_salt_key(monkeypatch)
    once = encrypt_callback_vars(_sample_metadata())
    twice = encrypt_callback_vars(once)
    assert once == twice


def test_encrypt_callback_vars_does_not_mutate_input(monkeypatch):
    _set_salt_key(monkeypatch)
    original = _sample_metadata()
    snapshot = copy.deepcopy(original)
    encrypt_callback_vars(original)
    assert original == snapshot


def test_decrypt_callback_vars_passes_through_legacy_plaintext(monkeypatch):
    _set_salt_key(monkeypatch)
    plaintext = _sample_metadata()
    decrypted = decrypt_callback_vars(plaintext)
    # legacy rows decrypt-fail and fall through unchanged
    assert (
        decrypted["logging"][0]["callback_vars"]["langfuse_secret_key"]
        == "sk-lf-secret"
    )


def test_callback_vars_helpers_handle_edge_shapes(monkeypatch):
    _set_salt_key(monkeypatch)
    assert encrypt_callback_vars(None) is None
    assert encrypt_callback_vars({}) == {}
    assert decrypt_callback_vars(None) is None
    assert decrypt_callback_vars({}) == {}

    # logging not a list / callback_vars not a dict — leave alone
    weird = {"logging": "not-a-list", "callback_settings": {"callback_vars": None}}
    assert encrypt_callback_vars(weird) == weird

    # empty/None callback_vars values stay as-is
    has_blanks = {
        "logging": [
            {
                "callback_vars": {
                    "langfuse_public_key": "",
                    "langfuse_secret_key": None,
                    "langfuse_host": "https://cloud.langfuse.com",
                }
            }
        ]
    }
    out = encrypt_callback_vars(has_blanks)
    cv = out["logging"][0]["callback_vars"]
    assert cv["langfuse_public_key"] == ""
    assert cv["langfuse_secret_key"] is None
    # langfuse_host is a routing field, not a credential — stays plain.
    assert cv["langfuse_host"] == "https://cloud.langfuse.com"


def test_encrypt_callback_vars_only_encrypts_credential_fields(monkeypatch):
    """Routing/identifier fields stay plaintext; credential fields encrypt."""
    _set_salt_key(monkeypatch)
    metadata = {
        "logging": [
            {
                "callback_vars": {
                    "langfuse_secret_key": "sk-real",
                    "langfuse_public_key": "pk-real",
                    "langfuse_host": "https://cloud.langfuse.com",
                    "langsmith_project": "my-proj",
                    "langsmith_base_url": "https://smith.example",
                    "gcs_path_service_account": "{json contents}",
                }
            }
        ]
    }
    cv = encrypt_callback_vars(metadata)["logging"][0]["callback_vars"]

    # Sensitive (key-name segments match SensitiveDataMasker patterns):
    assert cv["langfuse_secret_key"] != "sk-real"
    assert cv["langfuse_public_key"] != "pk-real"
    # Sensitive via the explicit gcs override:
    assert cv["gcs_path_service_account"] != "{json contents}"
    # Routing / identifiers stay plaintext:
    assert cv["langfuse_host"] == "https://cloud.langfuse.com"
    assert cv["langsmith_project"] == "my-proj"
    assert cv["langsmith_base_url"] == "https://smith.example"
