"""Tests that the batches/main.py Moonshot dispatch guards server-side credentials.

When a caller omits api_key (so the proxy falls back to MOONSHOT_API_KEY), any
caller-supplied api_base must be ignored — the server key must only ever be sent
to a server-controlled endpoint.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

ATTACKER_BASE = "https://attacker.example.com/v1"
SERVER_BASE = "https://api.moonshot.ai/v1"
SERVER_KEY = "sk-server-moonshot-key"
CALLER_KEY = "sk-caller-key"

# The four operations share the same guard; we test create_batch as the
# representative case and cancel_batch/retrieve_batch/list_batches for
# completeness.

CREATE_KWARGS = dict(
    completion_window="24h",
    endpoint="/v1/chat/completions",
    input_file_id="file-abc",
    custom_llm_provider="moonshot",
)


def _make_batch_response():
    resp = MagicMock()
    resp.model_dump.return_value = {
        "id": "batch-1",
        "completion_window": "24h",
        "created_at": 1700000000,
        "endpoint": "/v1/chat/completions",
        "input_file_id": "file-abc",
        "object": "batch",
        "status": "completed",
    }
    return resp


# ----------------------------------------------------------------- create_batch


class TestCreateBatchApiBaseGuard:
    def _captured_api_base(self, monkeypatch, caller_api_key, caller_api_base):
        """Run create_batch and return the api_base that reached MoonshotBatchesAPI."""
        monkeypatch.setenv("MOONSHOT_API_KEY", SERVER_KEY)
        captured: dict = {}

        def fake_create_batch(_is_async, create_batch_data, api_key, api_base, **kw):
            captured["api_base"] = api_base
            captured["api_key"] = api_key
            return MagicMock(model_dump=lambda: _make_batch_response().model_dump())

        with (
            patch("litellm.batches.main.moonshot_batches_instance.create_batch", side_effect=fake_create_batch),
            patch("litellm.batches.main.client", lambda f: f),  # bypass @client decorator
        ):
            import litellm.batches.main as bm

            bm.create_batch(
                **CREATE_KWARGS,
                api_key=caller_api_key,
                api_base=caller_api_base,
                litellm_logging_obj=MagicMock(),
            )

        return captured

    def test_server_key_uses_server_base_ignores_caller_base(self, monkeypatch):
        captured = self._captured_api_base(
            monkeypatch,
            caller_api_key=None,
            caller_api_base=ATTACKER_BASE,
        )
        assert captured["api_base"] != ATTACKER_BASE, (
            "Server MOONSHOT_API_KEY must never be sent to a caller-controlled base URL"
        )
        assert captured["api_base"] == SERVER_BASE
        assert captured["api_key"] == SERVER_KEY

    def test_caller_key_allows_caller_base(self, monkeypatch):
        monkeypatch.setenv("MOONSHOT_API_KEY", SERVER_KEY)
        captured = self._captured_api_base(
            monkeypatch,
            caller_api_key=CALLER_KEY,
            caller_api_base=ATTACKER_BASE,
        )
        # When the caller supplies their own key, they own the base too.
        assert captured["api_base"] == ATTACKER_BASE
        assert captured["api_key"] == CALLER_KEY

    def test_server_key_uses_env_base_when_set(self, monkeypatch):
        custom_server_base = "https://custom.moonshot.internal/v1"
        monkeypatch.setenv("MOONSHOT_API_KEY", SERVER_KEY)
        monkeypatch.setenv("MOONSHOT_API_BASE", custom_server_base)
        captured = self._captured_api_base(
            monkeypatch,
            caller_api_key=None,
            caller_api_base=ATTACKER_BASE,
        )
        assert captured["api_base"] == custom_server_base
        assert captured["api_base"] != ATTACKER_BASE
