import pytest
from pydantic import ValidationError

from litellm.proxy.control_plane.settings import ControlPlaneSettings, S3SinkSettings
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
    with pytest.raises(ValidationError, match="not https"):
        ControlPlaneSettings.model_validate(base)
    with pytest.raises(ValidationError, match="not https"):
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
