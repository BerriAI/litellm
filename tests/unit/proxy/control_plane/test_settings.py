import pytest
from pydantic import ValidationError

from litellm.proxy.control_plane.settings import (
    ControlPlaneSettings,
    S3SinkSettings,
    SentinelSettings,
    SplunkHecSettings,
)
from litellm.types.proxy.control_plane.auth import PASSWORD_MAX_LENGTH, USERNAME_MAX_LENGTH


@pytest.mark.parametrize(
    ("username", "password"),
    [("u" * (USERNAME_MAX_LENGTH + 1), "p"), ("u", "p" * (PASSWORD_MAX_LENGTH + 1))],
)
def test_configured_admin_credentials_share_the_login_request_bounds(username: str, password: str) -> None:
    with pytest.raises(ValidationError):
        ControlPlaneSettings.model_validate(
            {"LITELLM_MASTER_KEY": "sk-cp", "UI_USERNAME": username, "UI_PASSWORD": password}
        )


def test_the_s3_sink_refuses_to_start_without_a_bucket_and_reads_its_block_with_one() -> None:
    with pytest.raises(ValidationError, match="CONTROL_PLANE_AUDIT_S3_BUCKET"):
        ControlPlaneSettings.model_validate({"LITELLM_MASTER_KEY": "sk-m", "audit_sinks": ["log", "s3"]})
    settings = ControlPlaneSettings.model_validate(
        {
            "LITELLM_MASTER_KEY": "sk-m",
            "audit_sinks": ["s3"],
            "audit_s3_bucket": "audit",
            "audit_s3_prefix": "/cp/",
            "audit_s3_object_lock_mode": "COMPLIANCE",
            "audit_s3_retention_days": 7,
        }
    )
    assert settings.audit_s3 == S3SinkSettings("audit", "cp", None, None, "COMPLIANCE", 7, 30.0)
    assert ControlPlaneSettings.model_validate({"LITELLM_MASTER_KEY": "sk-m"}).audit_s3 is None


def test_a_plain_http_s3_endpoint_is_refused_unless_explicitly_allowed() -> None:
    base = {"LITELLM_MASTER_KEY": "sk-m", "audit_s3_bucket": "audit", "audit_s3_endpoint_url": "http://minio:9000"}
    with pytest.raises(ValidationError, match="CONTROL_PLANE_AUDIT_S3_ENDPOINT_URL is not https"):
        ControlPlaneSettings.model_validate(base)
    with pytest.raises(ValidationError, match="CONTROL_PLANE_AUDIT_S3_ALLOW_INSECURE_HTTP"):
        ControlPlaneSettings.model_validate({**base, "audit_s3_endpoint_url": "HTTP://minio:9000"})
    allowed = ControlPlaneSettings.model_validate({**base, "audit_s3_allow_insecure_http": True}).audit_s3
    assert allowed is not None and allowed.endpoint_url == "http://minio:9000"


def test_credentials_at_the_bound_are_accepted() -> None:
    settings = ControlPlaneSettings.model_validate(
        {
            "LITELLM_MASTER_KEY": "sk-cp",
            "UI_USERNAME": "u" * USERNAME_MAX_LENGTH,
            "UI_PASSWORD": "p" * PASSWORD_MAX_LENGTH,
        }
    )
    assert settings.ui_username is not None and len(settings.ui_username) == USERNAME_MAX_LENGTH


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
        ControlPlaneSettings.model_validate({"LITELLM_MASTER_KEY": "sk-m", field: value})
    if opt_out is not None:
        allowed = ControlPlaneSettings.model_validate({"LITELLM_MASTER_KEY": "sk-m", field: value, opt_out: True})
        assert getattr(allowed, field) == value


def test_the_splunk_and_sentinel_sinks_refuse_to_start_without_their_settings_and_read_them_when_set() -> None:
    with pytest.raises(ValidationError, match="CONTROL_PLANE_AUDIT_SPLUNK_URL, CONTROL_PLANE_AUDIT_SPLUNK_TOKEN"):
        ControlPlaneSettings.model_validate({"LITELLM_MASTER_KEY": "sk-m", "audit_sinks": ["splunk_hec"]})
    with pytest.raises(ValidationError, match="CONTROL_PLANE_AUDIT_SENTINEL_CLIENT_SECRET"):
        ControlPlaneSettings.model_validate(
            {
                "LITELLM_MASTER_KEY": "sk-m",
                "audit_sinks": ["sentinel"],
                "audit_sentinel_endpoint": "https://dce.test",
                "audit_sentinel_dcr_immutable_id": "dcr",
                "audit_sentinel_tenant_id": "t",
                "audit_sentinel_client_id": "c",
            }
        )
    settings = ControlPlaneSettings.model_validate(
        {
            "LITELLM_MASTER_KEY": "sk-m",
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
    splunk = settings.audit_splunk
    sentinel = settings.audit_sentinel
    assert splunk is not None and (splunk.url, splunk.token.get_secret_value(), splunk.index) == (
        "https://hec.test:8088",
        "tok",
        None,
    )
    assert sentinel is not None and sentinel.authority_host == "https://login.microsoftonline.us"
    assert (
        sentinel.stream_name == "Custom-LiteLLMControlPlaneAudit_CL"
        and sentinel.client_secret.get_secret_value() == "s"
    )
