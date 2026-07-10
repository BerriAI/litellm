"""Unit coverage for batch credential env helpers (no live proxy)."""

from __future__ import annotations

import pytest

import provider_env


def test_is_credential_error_body_matches_known_gateway_messages() -> None:
    assert provider_env.is_credential_error_body(
        '{"error":{"message":"GCS bucket_name is required"}}'
    )
    assert provider_env.is_credential_error_body(
        "S3 bucket_name is required. Set 's3_bucket_name' in litellm_params"
    )
    assert provider_env.is_credential_error_body(
        "Your default credentials were not found. To set up Application Default Credentials"
    )
    assert not provider_env.is_credential_error_body(
        "Filtering by 'provider' is not supported when using managed batches."
    )


def test_missing_env_for_provider_reports_bedrock_bucket_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_REGION",
        "AWS_BATCH_ROLE_ARN",
        "AWS_BATCH_S3_BUCKET",
        "AWS_S3_BUCKET_NAME",
    ):
        monkeypatch.delenv(name, raising=False)

    missing = provider_env.missing_env_for_provider("bedrock")
    assert "AWS_ACCESS_KEY_ID" in missing
    assert "AWS_BATCH_S3_BUCKET|AWS_S3_BUCKET_NAME" in missing

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "ak")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "sk")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("AWS_BATCH_ROLE_ARN", "arn:aws:iam::1:role/r")
    monkeypatch.setenv("AWS_S3_BUCKET_NAME", "b")

    assert provider_env.missing_env_for_provider("bedrock") == ()
