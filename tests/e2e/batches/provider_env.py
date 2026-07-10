"""Env vars the gateway must have for each batch provider (Docker .env or K8s secrets).

Deployments register with ``os.environ/NAME`` refs; the proxy resolves them from
*its* process environment. Locally that is docker compose ``env_file: .env``;
on EKS it is the secret store mounted into the gateway pods. The e2e runner
checks the same names so a missing secret becomes a skip, not a red failure.
"""

from __future__ import annotations

import os
from typing import Mapping

PROVIDER_REQUIRED_ENV: Mapping[str, tuple[str, ...]] = {
    "openai": ("OPENAI_API_KEY",),
    "azure": ("AZURE_API_KEY", "AZURE_API_BASE"),
    "vertex_ai": (
        "VERTEXAI_PROJECT",
        "VERTEXAI_CREDENTIALS",
        "GCS_BUCKET_NAME",
    ),
    "bedrock": (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_REGION",
        "AWS_BATCH_ROLE_ARN",
    ),
}

BEDROCK_BUCKET_ENV: tuple[str, ...] = ("AWS_BATCH_S3_BUCKET", "AWS_S3_BUCKET_NAME")

CREDENTIAL_ERROR_MARKERS: tuple[str, ...] = (
    "gcs bucket_name is required",
    "s3 bucket_name is required",
    "bucket_name is required",
    "default credentials were not found",
    "application default credentials",
    "aws iam role arn is required",
    "missing mistral api key",
    "no key is set either in the environment",
    "openai_api_key not set",
    "authentication error, invalid",
    "incorrect api key provided",
    "could not resolve authentication",
)


def _present(name: str) -> bool:
    value = os.environ.get(name)
    return value is not None and value.strip() != ""


def missing_env_for_provider(provider: str) -> tuple[str, ...]:
    required = PROVIDER_REQUIRED_ENV.get(provider, ())
    missing = tuple(name for name in required if not _present(name))
    if provider == "bedrock" and not any(_present(name) for name in BEDROCK_BUCKET_ENV):
        missing = (*missing, "AWS_BATCH_S3_BUCKET|AWS_S3_BUCKET_NAME")
    return missing


def skip_reason_missing_env(provider: str) -> str | None:
    missing = missing_env_for_provider(provider)
    if not missing:
        return None
    return (
        f"batch provider {provider!r} missing env on runner "
        f"(gateway needs the same via docker .env or K8s secrets): {', '.join(missing)}"
    )


def is_credential_error_body(body: str) -> bool:
    lowered = body.lower()
    return any(marker in lowered for marker in CREDENTIAL_ERROR_MARKERS)
