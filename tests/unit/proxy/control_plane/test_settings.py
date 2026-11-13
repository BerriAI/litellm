from typing import Final

import pytest
from pydantic import ValidationError

from litellm.proxy.control_plane.audit.event import CONTROL_PLANE_ID
from litellm.proxy.control_plane.settings import (
    ControlPlaneSettings,
    S3SinkSettings,
    load_settings,
)
from litellm.types.proxy.control_plane.auth import PASSWORD_MAX_LENGTH, USERNAME_MAX_LENGTH

MASTER_KEY: Final = "01234567890123456789012345678901"


@pytest.mark.parametrize(
    ("username", "password"),
    [("u" * (USERNAME_MAX_LENGTH + 1), "p"), ("u", "p" * (PASSWORD_MAX_LENGTH + 1))],
)
def test_configured_admin_credentials_share_the_login_request_bounds(username: str, password: str) -> None:
    with pytest.raises(ValidationError):
        ControlPlaneSettings.model_validate(
            {"LITELLM_MASTER_KEY": MASTER_KEY, "UI_USERNAME": username, "UI_PASSWORD": password}
        )


def test_password_login_without_default_roles_requires_a_custom_admin_role() -> None:
    with pytest.raises(ValidationError, match="custom admin role"):
        ControlPlaneSettings.model_validate(
            {
                "LITELLM_MASTER_KEY": MASTER_KEY,
                "UI_USERNAME": "admin",
                "UI_PASSWORD": "pw",
                "include_default_roles": False,
            }
        )
    settings: Final = ControlPlaneSettings.model_validate(
        {
            "LITELLM_MASTER_KEY": MASTER_KEY,
            "UI_USERNAME": "admin",
            "UI_PASSWORD": "pw",
            "include_default_roles": False,
            "roles": {"admin": {"scopes": ["dataplanes:read"]}},
        }
    )
    assert not settings.include_default_roles and "admin" in settings.roles


def test_the_s3_sink_refuses_to_start_without_a_bucket_and_reads_its_block_with_one() -> None:
    with pytest.raises(ValidationError, match="CONTROL_PLANE_AUDIT_S3_BUCKET"):
        ControlPlaneSettings.model_validate({"LITELLM_MASTER_KEY": MASTER_KEY, "audit_sinks": ["log", "s3"]})
    settings: Final = ControlPlaneSettings.model_validate(
        {
            "LITELLM_MASTER_KEY": MASTER_KEY,
            "audit_sinks": ["s3"],
            "audit_s3_bucket": "audit",
            "audit_s3_prefix": "/cp/",
            "audit_s3_object_lock_mode": "COMPLIANCE",
            "audit_s3_retention_days": 7,
        }
    )
    assert settings.audit_s3 == S3SinkSettings("audit", "cp", None, None, "COMPLIANCE", 7, 30.0)
    assert ControlPlaneSettings.model_validate({"LITELLM_MASTER_KEY": MASTER_KEY}).audit_s3 is None


@pytest.mark.parametrize("bucket", ["", " \t "])
def test_the_s3_sink_rejects_a_blank_bucket(bucket: str) -> None:
    with pytest.raises(ValidationError, match="must not be empty or whitespace"):
        ControlPlaneSettings.model_validate(
            {"LITELLM_MASTER_KEY": MASTER_KEY, "audit_sinks": ["s3"], "audit_s3_bucket": bucket}
        )


def test_a_plain_http_s3_endpoint_is_refused_unless_explicitly_allowed() -> None:
    base: Final = {
        "LITELLM_MASTER_KEY": MASTER_KEY,
        "audit_s3_bucket": "audit",
        "audit_s3_endpoint_url": "http://minio:9000",
    }
    with pytest.raises(ValidationError, match="CONTROL_PLANE_AUDIT_S3_ENDPOINT_URL is not https"):
        ControlPlaneSettings.model_validate(base)
    with pytest.raises(ValidationError, match="CONTROL_PLANE_AUDIT_S3_ALLOW_INSECURE_HTTP"):
        ControlPlaneSettings.model_validate({**base, "audit_s3_endpoint_url": "HTTP://minio:9000"})
    allowed: Final = ControlPlaneSettings.model_validate({**base, "audit_s3_allow_insecure_http": True}).audit_s3
    assert allowed is not None and allowed.endpoint_url == "http://minio:9000"


def test_credentials_at_the_bound_are_accepted() -> None:
    settings: Final = ControlPlaneSettings.model_validate(
        {
            "LITELLM_MASTER_KEY": MASTER_KEY,
            "UI_USERNAME": "u" * USERNAME_MAX_LENGTH,
            "UI_PASSWORD": "p" * PASSWORD_MAX_LENGTH,
        }
    )
    assert settings.ui_username is not None and len(settings.ui_username) == USERNAME_MAX_LENGTH


def test_dataplanes_environment_and_file_cannot_both_be_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_MASTER_KEY", MASTER_KEY)
    monkeypatch.setenv("CONTROL_PLANE_DATAPLANES", "{}")
    monkeypatch.setenv("CONTROL_PLANE_DATAPLANES_FILE", "dataplanes.yaml")

    with pytest.raises(ValueError, match="CONTROL_PLANE_DATAPLANES and CONTROL_PLANE_DATAPLANES_FILE"):
        load_settings()


def test_fanout_concurrency_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        ControlPlaneSettings.model_validate({"LITELLM_MASTER_KEY": MASTER_KEY, "fanout_concurrency": 0})


def test_master_key_minimum_length_uses_utf8_bytes() -> None:
    with pytest.raises(ValidationError):
        ControlPlaneSettings.model_validate({"LITELLM_MASTER_KEY": "é" * 15 + "x"})
    settings: Final = ControlPlaneSettings.model_validate({"LITELLM_MASTER_KEY": "é" * 16})
    assert len(settings.master_key.get_secret_value().encode("utf-8")) == 32


def test_usage_window_maximum_is_configurable() -> None:
    settings: Final = ControlPlaneSettings.model_validate(
        {"LITELLM_MASTER_KEY": MASTER_KEY, "usage_max_window_days": 12}
    )
    assert settings.usage_max_window_days == 12


def test_dataplane_id_cannot_be_reserved_for_the_control_plane() -> None:
    with pytest.raises(ValidationError, match="dataplane id 'control-plane' is reserved"):
        ControlPlaneSettings.model_validate(
            {
                "LITELLM_MASTER_KEY": MASTER_KEY,
                "dataplanes": {CONTROL_PLANE_ID: {"url": "https://control.test"}},
            }
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("audit_capacity", 0),
        ("audit_queue_size", 1),
        ("audit_batch_size", 0),
        ("audit_drain_seconds", 0),
    ],
)
def test_audit_capacity_and_delivery_limits_must_be_positive(field: str, value: int) -> None:
    with pytest.raises(ValidationError):
        ControlPlaneSettings.model_validate({"LITELLM_MASTER_KEY": MASTER_KEY, field: value})


def test_audit_queue_size_allows_two_slots() -> None:
    settings: Final = ControlPlaneSettings.model_validate({"LITELLM_MASTER_KEY": MASTER_KEY, "audit_queue_size": 2})
    assert settings.audit_queue_size == 2


def test_strict_audit_requires_at_least_one_sink() -> None:
    with pytest.raises(ValidationError, match="requires at least one audit sink"):
        ControlPlaneSettings.model_validate({"LITELLM_MASTER_KEY": MASTER_KEY, "audit_strict": True, "audit_sinks": ()})


@pytest.mark.parametrize(
    ("field", "value", "opt_out"),
    [
        ("audit_splunk_url", "http://hec.internal:8088", "audit_splunk_allow_insecure_http"),
        ("audit_sentinel_endpoint", "HTTP://dce.test", None),
        ("audit_sentinel_authority_host", "http://login.test", None),
    ],
)
def test_sink_urls_that_carry_a_credential_must_be_https(field: str, value: str, opt_out: str | None) -> None:
    with pytest.raises(ValidationError, match=f"CONTROL_PLANE_{field.upper()} is not https"):
        ControlPlaneSettings.model_validate({"LITELLM_MASTER_KEY": MASTER_KEY, field: value})
    if opt_out is not None:
        allowed: Final = ControlPlaneSettings.model_validate(
            {"LITELLM_MASTER_KEY": MASTER_KEY, field: value, opt_out: True}
        )
        assert getattr(allowed, field) == value


def test_the_splunk_and_sentinel_sinks_refuse_to_start_without_their_settings_and_read_them_when_set() -> None:
    with pytest.raises(ValidationError, match="CONTROL_PLANE_AUDIT_SPLUNK_URL, CONTROL_PLANE_AUDIT_SPLUNK_TOKEN"):
        ControlPlaneSettings.model_validate({"LITELLM_MASTER_KEY": MASTER_KEY, "audit_sinks": ["splunk_hec"]})
    with pytest.raises(ValidationError, match="CONTROL_PLANE_AUDIT_SENTINEL_CLIENT_SECRET"):
        ControlPlaneSettings.model_validate(
            {
                "LITELLM_MASTER_KEY": MASTER_KEY,
                "audit_sinks": ["sentinel"],
                "audit_sentinel_endpoint": "https://dce.test",
                "audit_sentinel_dcr_immutable_id": "dcr",
                "audit_sentinel_tenant_id": "t",
                "audit_sentinel_client_id": "c",
            }
        )
    settings: Final = ControlPlaneSettings.model_validate(
        {
            "LITELLM_MASTER_KEY": MASTER_KEY,
            "audit_sinks": ["splunk_hec", "sentinel"],
            "audit_splunk_url": "https://hec.test:8088",
            "audit_splunk_token": "tok",
            "audit_sentinel_endpoint": "https://dce.test",
            "audit_sentinel_dcr_immutable_id": "dcr",
            "audit_sentinel_tenant_id": "t",
            "audit_sentinel_client_id": "c",
            "audit_sentinel_client_secret": "s",
            "audit_sentinel_authority_host": "https://login.microsoftonline.us/",
        }
    )
    splunk: Final = settings.audit_splunk
    sentinel: Final = settings.audit_sentinel
    assert splunk is not None and (splunk.url, splunk.token.get_secret_value(), splunk.index) == (
        "https://hec.test:8088",
        "tok",
        None,
    )
    assert sentinel is not None and sentinel.authority_host == "https://login.microsoftonline.us"
    assert (
        sentinel.stream_name == "Custom-LiteLLMControlPlaneAudit_CL"
        and sentinel.client_secret.get_secret_value() == "s"
        and sentinel.token_scope == "https://monitor.azure.us/.default"
    )
    china: Final = settings.model_copy(
        update={"audit_sentinel_authority_host": "https://login.chinacloudapi.cn/"}
    ).audit_sentinel
    public: Final = settings.model_copy(
        update={"audit_sentinel_authority_host": "https://login.microsoftonline.com/"}
    ).audit_sentinel
    override: Final = settings.model_copy(
        update={"audit_sentinel_token_scope": "https://monitor.override.test/.default"}
    ).audit_sentinel
    assert china is not None and china.token_scope == "https://monitor.azure.cn/.default"
    assert public is not None and public.token_scope == "https://monitor.azure.com/.default"
    assert override is not None and override.token_scope == "https://monitor.override.test/.default"


@pytest.mark.parametrize(
    ("field", "blank"),
    [
        ("audit_splunk_url", ""),
        ("audit_splunk_url", " \t "),
        ("audit_splunk_token", ""),
        ("audit_splunk_token", " \t "),
        ("audit_splunk_index", ""),
        ("audit_splunk_index", " \t "),
        ("audit_splunk_source", ""),
        ("audit_splunk_source", " \t "),
        ("audit_sentinel_endpoint", ""),
        ("audit_sentinel_endpoint", " \t "),
        ("audit_sentinel_dcr_immutable_id", ""),
        ("audit_sentinel_dcr_immutable_id", " \t "),
        ("audit_sentinel_stream_name", ""),
        ("audit_sentinel_stream_name", "  "),
        ("audit_sentinel_tenant_id", ""),
        ("audit_sentinel_tenant_id", " \t "),
        ("audit_sentinel_client_id", ""),
        ("audit_sentinel_client_id", " \t "),
        ("audit_sentinel_client_secret", ""),
        ("audit_sentinel_client_secret", " \t "),
        ("audit_sentinel_token_scope", ""),
        ("audit_sentinel_token_scope", " \t "),
    ],
)
def test_sink_strings_reject_empty_or_whitespace(field: str, blank: str) -> None:
    settings: Final = {
        "LITELLM_MASTER_KEY": MASTER_KEY,
        "audit_sinks": ["splunk_hec", "sentinel"],
        "audit_splunk_url": "https://hec.test:8088",
        "audit_splunk_token": "token",
        "audit_sentinel_endpoint": "https://dce.test",
        "audit_sentinel_dcr_immutable_id": "dcr",
        "audit_sentinel_tenant_id": "tenant",
        "audit_sentinel_client_id": "client",
        "audit_sentinel_client_secret": "secret",
    }
    with pytest.raises(ValidationError, match="must not be empty or whitespace"):
        ControlPlaneSettings.model_validate({**settings, field: blank})


@pytest.mark.parametrize("index", [None, "audit"])
def test_splunk_index_is_optional_but_not_blank_when_set(index: str | None) -> None:
    settings: Final = ControlPlaneSettings.model_validate(
        {
            "LITELLM_MASTER_KEY": MASTER_KEY,
            "audit_splunk_url": "https://hec.test:8088",
            "audit_splunk_token": "token",
            "audit_splunk_index": index,
        }
    )
    splunk: Final = settings.audit_splunk
    assert splunk is not None and splunk.index == index
