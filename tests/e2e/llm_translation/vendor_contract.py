"""Shared helpers for vendor API contract e2e tests (LIT-4778).

Status-centric assertions used across endpoint negatives, sanitization, and
auth matrix cases so each test file stays thin.
"""

from __future__ import annotations

from e2e_http import StreamingResponse


def is_client_error(status: int) -> bool:
    return 400 <= status < 500


def is_auth_denied(status: int) -> bool:
    return status in (401, 403)


def assert_not_server_error(result: StreamingResponse, context: str) -> None:
    assert result.status_code not in (500, 502, 503), (
        f"{context}: proxy must not 5xx, got {result.status_code}: {result.body[:300]}"
    )


def assert_client_error(result: StreamingResponse, context: str) -> None:
    assert is_client_error(result.status_code), (
        f"{context}: expected 4xx, got {result.status_code}: {result.body[:300]}"
    )


def assert_error_or_server_known(result: StreamingResponse, context: str) -> None:
    """Missing required fields may be 4xx or 5xx per known acceptable proxy behavior."""
    assert result.status_code in range(400, 600), (
        f"{context}: expected error status, got {result.status_code}: {result.body[:300]}"
    )
    assert result.status_code != 200


def assert_auth_denied(result: StreamingResponse, context: str) -> None:
    assert is_auth_denied(result.status_code), (
        f"{context}: expected 401/403, got {result.status_code}: {result.body[:300]}"
    )
