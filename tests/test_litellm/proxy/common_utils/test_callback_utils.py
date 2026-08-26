import copy
import sys
from types import ModuleType, SimpleNamespace

import pytest


from litellm.proxy.common_utils.callback_utils import (
    add_guardrail_scan_id,
    add_policy_to_applied_policies_header,
    decrypt_callback_vars,
    encrypt_callback_vars,
    get_logging_caching_headers,
    initialize_callbacks_on_proxy,
    get_remaining_tokens_and_requests_from_request_data,
    normalize_callback_names,
    sanitize_openai_provider_metadata,
    strip_callback_config,
)
import litellm
from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.utils import ProxyLogging

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


@patch(
    "litellm.proxy.common_utils.callback_utils.CustomLogger.get_callback_env_vars",
    return_value=["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"],
)
def test_process_callback_falls_back_to_process_env(mock_get_env_vars, monkeypatch):
    """A callback env var set only in the process env must be surfaced.

    The logging integrations read their config from the process environment, so a
    callback configured purely via env vars (IaC) is live even with no stored
    entry. Reporting it as unset makes a working callback read as unconfigured.
    """
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "env-public-key")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "env-secret-key")
    # stored config only carries the public key; the secret is env-only
    environment_variables = {"LANGFUSE_PUBLIC_KEY": "db-public-key"}

    result = process_callback(
        _callback="langfuse",
        callback_type="success",
        environment_variables=environment_variables,
    )

    # stored value wins; the env-only var is resolved rather than reported None
    assert result["variables"] == {
        "LANGFUSE_PUBLIC_KEY": "db-public-key",
        "LANGFUSE_SECRET_KEY": "env-secret-key",
    }


@patch(
    "litellm.proxy.common_utils.callback_utils.CustomLogger.get_callback_env_vars",
    return_value=["LANGFUSE_SECRET_KEY"],
)
def test_process_callback_reports_none_when_absent_everywhere(mock_get_env_vars, monkeypatch):
    """A var set in neither the stored config nor the process env stays None."""
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    result = process_callback(
        _callback="langfuse",
        callback_type="success",
        environment_variables={},
    )

    assert result["variables"] == {"LANGFUSE_SECRET_KEY": None}


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


def test_add_guardrail_scan_id_dedupes_and_becomes_response_header():
    request_data = {"litellm_metadata": {}}

    add_guardrail_scan_id(request_data=request_data, scan_id="scan-1")
    add_guardrail_scan_id(request_data=request_data, scan_id="scan-1")
    add_guardrail_scan_id(request_data=request_data, scan_id="scan-2")
    add_guardrail_scan_id(request_data=request_data, scan_id=None)

    assert request_data["litellm_metadata"]["guardrail_scan_ids"] == ("scan-1", "scan-2")
    assert get_logging_caching_headers(request_data)["x-litellm-guardrail-scan-id"] == "scan-1,scan-2"


def test_get_logging_caching_headers_omits_scan_id_header_without_scans():
    assert "x-litellm-guardrail-scan-id" not in get_logging_caching_headers({"litellm_metadata": {}})


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


def test_initialize_callbacks_on_proxy_lakera_ignores_non_dict_callback_settings(
    monkeypatch,
):
    """Regression: a non-dict value under callback_settings.lakera_prompt_injection
    must not crash initialize_callbacks_on_proxy.

    Forwarding callback_settings as callback_specific_params (so callbacks like
    DatadogCostManagementLogger receive their init params) exposes the lakera
    branch, which previously did lakeraAI_Moderation(**callback_specific_params[
    "lakera_prompt_injection"]) with no isinstance(dict) guard. For a config like
    {"lakera_prompt_injection": "x"} that is `**"x"` -> TypeError: argument after
    ** must be a mapping, not str. The branch now guards on isinstance(dict),
    matching the presidio / datadog_cost_management branches.
    """
    captured = {}

    class _DummyLakera:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

    # Inject a fake lakera_ai module so the branch's
    # `from ...lakera_ai import lakeraAI_Moderation` resolves to our stub without
    # importing the real module (which imports proxy_server symbols not present
    # under the stubbed proxy_server below).
    fake_lakera = ModuleType("litellm.proxy.guardrails.guardrail_hooks.lakera_ai")
    fake_lakera.lakeraAI_Moderation = _DummyLakera
    monkeypatch.setitem(
        sys.modules,
        "litellm.proxy.guardrails.guardrail_hooks.lakera_ai",
        fake_lakera,
    )
    monkeypatch.setitem(
        sys.modules,
        "litellm.proxy.proxy_server",
        SimpleNamespace(prisma_client=None),
    )

    original_callbacks = (
        list(litellm.callbacks) if isinstance(litellm.callbacks, list) else []
    )
    litellm.callbacks = []
    try:
        # A non-dict value must be ignored (init_params stays {}), not **-unpacked.
        initialize_callbacks_on_proxy(
            value=["lakera_prompt_injection"],
            premium_user=False,
            config_file_path=".",
            litellm_settings={},
            callback_specific_params={"lakera_prompt_injection": "any-string"},
        )
        assert captured["kwargs"] == {}
        assert any(isinstance(c, _DummyLakera) for c in litellm.callbacks)
    finally:
        litellm.callbacks = original_callbacks


@pytest.mark.parametrize("bad_root", [None, True])
def test_initialize_callbacks_on_proxy_non_dict_callback_specific_params_root(
    monkeypatch, bad_root
):
    """Regression: a blank `callback_settings:` key in YAML loads as None (and
    `callback_settings: true` as a bool); load_config forwards that value
    verbatim as callback_specific_params. Membership tests like
    `"compression_interception" in callback_specific_params` then raise
    TypeError and abort proxy startup. A non-dict root must be normalized to {}
    so the callback initializes with its defaults.
    """
    monkeypatch.setitem(
        sys.modules,
        "litellm.proxy.proxy_server",
        SimpleNamespace(prisma_client=None),
    )
    from litellm.integrations.compression_interception.handler import (
        CompressionInterceptionLogger,
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
            litellm_settings={},
            callback_specific_params=bad_root,
        )
        assert any(
            isinstance(c, CompressionInterceptionLogger) for c in litellm.callbacks
        )
    finally:
        litellm.callbacks = original_callbacks


def test_strip_callback_config_drops_credential_bearing_slots():
    """
    `logging` and `callback_settings` hold operator-configured integration
    credentials. Both must be dropped from the key/team metadata the proxy
    stamps into request metadata, while every other field survives untouched
    (`priority` is read back by the dynamic rate limiter, `guardrails` by the
    guardrail hooks).
    """
    metadata = {
        "logging": [
            {
                "callback_name": "langsmith",
                "callback_vars": {"langsmith_api_key": "litellm_enc::ciphertext"},
            }
        ],
        "callback_settings": {"callback_vars": {"langfuse_secret_key": "litellm_enc::other"}},
        "secret_manager_settings": {"vault_token": "vt-secret"},
        "priority": "high",
        "guardrails": ["presidio"],
        "langsmith_provisioning": {"api_key_id": "prov-1"},
    }

    stripped = strip_callback_config(metadata)

    assert "logging" not in stripped
    assert "callback_settings" not in stripped
    assert "secret_manager_settings" not in stripped
    assert stripped["priority"] == "high"
    assert stripped["guardrails"] == ["presidio"]
    assert stripped["langsmith_provisioning"] == {"api_key_id": "prov-1"}
    # the caller's dict (UserAPIKeyAuth.metadata) is shared state - never mutate it
    assert "logging" in metadata
    assert "callback_settings" in metadata


@pytest.mark.parametrize("value", [None, "not-a-dict", 42])
def test_strip_callback_config_passes_through_non_dicts(value):
    assert strip_callback_config(value) is value


# ---------------------------------------------------------------------------
# initialize_callbacks_on_proxy: dotted-path entries must resolve to something
# the request path can actually dispatch
# ---------------------------------------------------------------------------

_PROBE_MODULE_NAME = "custom_callback_probe"

_PROBE_MODULE_SOURCE = '''
from litellm.integrations.custom_logger import CustomLogger


class FloorMaxTokens(CustomLogger):
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        data["max_tokens"] = 16
        return data


class NotALogger:
    pass


def log_event_fn(kwargs, response_obj, start_time, end_time):
    return None


NOT_A_CALLBACK = "some-plain-string"

proxy_handler_instance = FloorMaxTokens()
'''


@pytest.fixture
def probe_config_path(tmp_path):
    """Write a callback module next to a config.yaml, the layout get_instance_fn's file
    branch expects, and restore every global the load + dispatch path touches.

    ``ProxyLogging._callback_capabilities_cache`` is keyed on the id()s of the
    litellm.callbacks members, so an entry left behind here can be read back by an
    unrelated test whose (len, ids) signature happens to collide.
    """
    (tmp_path / f"{_PROBE_MODULE_NAME}.py").write_text(_PROBE_MODULE_SOURCE)

    original_callbacks = (
        list(litellm.callbacks) if isinstance(litellm.callbacks, list) else litellm.callbacks
    )
    litellm.callbacks = []
    ProxyLogging._callback_capabilities_cache.clear()
    try:
        yield str(tmp_path / "config.yaml")
    finally:
        litellm.callbacks = original_callbacks
        ProxyLogging._callback_capabilities_cache.clear()


def _load_callbacks(value, config_file_path):
    initialize_callbacks_on_proxy(
        value=value,
        premium_user=False,
        config_file_path=config_file_path,
        litellm_settings={},
        callback_specific_params={},
    )


def test_initialize_callbacks_on_proxy_rejects_class_valued_entry(probe_config_path):
    """A class path loads an object that fails the `isinstance(_callback, CustomLogger)`
    dispatch gate in ProxyLogging.pre_call_hook, so the proxy used to boot clean and
    silently never run the hook. Config load must fail instead."""
    entry = f"{_PROBE_MODULE_NAME}.FloorMaxTokens"

    with pytest.raises(ValueError, match='litellm_settings\\.callbacks entry') as exc_info:
        _load_callbacks([entry], probe_config_path)

    message = str(exc_info.value)
    assert entry in message
    assert "the class" in message
    assert "FloorMaxTokens" in message
    assert f"{_PROBE_MODULE_NAME}.proxy_handler_instance" in message
    assert litellm.callbacks == []


@pytest.mark.parametrize(
    "attribute, expected_fragment",
    [
        ("NotALogger", "the class"),
        ("NOT_A_CALLBACK", "str 'some-plain-string'"),
    ],
)
def test_initialize_callbacks_on_proxy_rejects_non_dispatchable_values(
    probe_config_path, attribute, expected_fragment
):
    entry = f"{_PROBE_MODULE_NAME}.{attribute}"

    with pytest.raises(ValueError, match='litellm_settings\\.callbacks entry') as exc_info:
        _load_callbacks([entry], probe_config_path)

    message = str(exc_info.value)
    assert entry in message
    assert expected_fragment in message
    assert litellm.callbacks == []


def test_initialize_callbacks_on_proxy_rejects_class_valued_non_list_value(probe_config_path):
    entry = f"{_PROBE_MODULE_NAME}.FloorMaxTokens"

    with pytest.raises(ValueError, match='litellm_settings\\.callbacks entry') as exc_info:
        _load_callbacks(entry, probe_config_path)

    assert entry in str(exc_info.value)


@pytest.mark.asyncio
async def test_initialize_callbacks_on_proxy_instance_entry_runs_pre_call_hook(probe_config_path):
    """Positive control: the supported shape must still load AND still run. Drives the
    real ProxyLogging.pre_call_hook, which is where a class-valued entry goes silent."""
    _load_callbacks([f"{_PROBE_MODULE_NAME}.proxy_handler_instance"], probe_config_path)

    assert len(litellm.callbacks) == 1
    assert isinstance(litellm.callbacks[0], CustomLogger)

    ProxyLogging._callback_capabilities_cache.clear()
    proxy_logging = ProxyLogging(user_api_key_cache=DualCache())
    data = await proxy_logging.pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(api_key="sk-probe"),
        data={
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 1,
            "metadata": {},
        },
        call_type="acompletion",
    )

    assert data["max_tokens"] == 16


def test_initialize_callbacks_on_proxy_keeps_known_string_callback(probe_config_path):
    """Non-narrowing control: a known callback name never reaches get_instance_fn and
    stays a plain string in litellm.callbacks."""
    _load_callbacks(["langfuse"], probe_config_path)

    assert litellm.callbacks == ["langfuse"]


def test_initialize_callbacks_on_proxy_accepts_plain_function_callback(probe_config_path):
    """Non-narrowing control: litellm.callbacks is typed
    `Callable | <known name> | CustomLogger`, so a dotted path resolving to a plain
    function is a supported shape and must keep loading."""
    _load_callbacks([f"{_PROBE_MODULE_NAME}.log_event_fn"], probe_config_path)

    assert [getattr(cb, "__name__", None) for cb in litellm.callbacks] == ["log_event_fn"]


def test_initialize_callbacks_on_proxy_accepts_instance_non_list_value(probe_config_path):
    _load_callbacks(f"{_PROBE_MODULE_NAME}.proxy_handler_instance", probe_config_path)

    assert len(litellm.callbacks) == 1
    assert isinstance(litellm.callbacks[0], CustomLogger)
