"""
Shared VCR configuration for ``tests/llm_translation``.

This module centralises the cassette setup used by tests that would otherwise
hit a real LLM provider over the network. The goal is to let CI replay
recorded HTTP traffic by default — no API keys required — and to provide a
single switch for re-recording cassettes against the live provider.

Usage in a test::

    from .vcr_config import litellm_vcr  # noqa: E402

    @litellm_vcr.use_cassette("anthropic_basic_completion.yaml")
    def test_basic_completion():
        resp = litellm.completion(
            model="anthropic/claude-sonnet-4-5-20250929",
            messages=[{"role": "user", "content": "Hello!"}],
        )
        assert resp.choices[0].message.content

Recording mode
--------------
By default the cassette is replayed (``record_mode='none'``). To re-record:

    LITELLM_VCR_RECORD_MODE=once \\
        ANTHROPIC_API_KEY=sk-ant-... \\
        uv run pytest tests/llm_translation/test_anthropic_completion_vcr.py

Valid values for ``LITELLM_VCR_RECORD_MODE`` mirror vcrpy's record modes:
``none`` (replay only — fail on missing cassette), ``once`` (record if the
cassette doesn't exist), ``new_episodes`` (append new interactions), and
``all`` (always re-record). See the vcrpy docs for details.

Why this exists
---------------
Per the discussion that produced LIT-2683, our e2e tests repeatedly drained
provider billing accounts and produced flaky CI on outages. Recording the
HTTP exchange once and replaying it on subsequent runs gives us realistic
provider responses (including streaming, headers, and edge-case payloads)
without per-PR cost or rate-limit risk. Re-record periodically to catch
real provider drift.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import vcr

CASSETTE_DIR: Path = Path(__file__).parent / "cassettes"

# Headers that must never be persisted to a cassette. These are matched
# case-insensitively by vcrpy.
_FILTERED_REQUEST_HEADERS = (
    "authorization",
    "x-api-key",
    "anthropic-api-key",
    "openai-api-key",
    "azure-api-key",
    "api-key",
    "cookie",
    "x-amz-security-token",
    "x-amz-date",
    "x-amz-content-sha256",
    "amz-sdk-invocation-id",
    "amz-sdk-request",
)

_FILTERED_RESPONSE_HEADERS = (
    "set-cookie",
    "x-request-id",
    "cf-ray",
    "anthropic-organization-id",
    "openai-organization",
    "request-id",
)


def _record_mode() -> str:
    """Resolve the active vcrpy record mode from the environment.

    Defaults to ``"none"`` so CI never accidentally hits the live provider.
    """
    mode = os.environ.get("LITELLM_VCR_RECORD_MODE", "none").strip().lower()
    if mode not in {"none", "once", "new_episodes", "all"}:
        raise ValueError(
            f"LITELLM_VCR_RECORD_MODE={mode!r} is not a valid vcrpy record mode."
        )
    return mode


def _build_vcr() -> vcr.VCR:
    """Construct the shared ``VCR`` instance used by translation tests."""
    return vcr.VCR(
        cassette_library_dir=str(CASSETTE_DIR),
        record_mode=_record_mode(),
        # Match on method + URI + body so streaming vs non-streaming and
        # different prompts get distinct cassettes.
        match_on=("method", "scheme", "host", "port", "path", "query", "body"),
        filter_headers=list(_FILTERED_REQUEST_HEADERS),
        decode_compressed_response=True,
    )


def _scrub_response(response: Any) -> Any:
    """Strip per-request response headers we don't want in the cassette."""
    if not isinstance(response, dict):
        return response
    headers = response.get("headers") or {}
    if isinstance(headers, dict):
        for header in list(headers):
            if header.lower() in _FILTERED_RESPONSE_HEADERS:
                headers.pop(header, None)
    return response


litellm_vcr: vcr.VCR = _build_vcr()
litellm_vcr.before_record_response = _scrub_response


__all__ = ["litellm_vcr", "CASSETTE_DIR"]
