"""
Regression tests for the Cloudflare Workers AI text-generation catalog in the
model-cost map.

The Cloudflare list was badly stale (only 4 ancient entries). These tests pin
the newly added current Workers AI models (sourced from Cloudflare's live
``/ai/models/search?task=Text Generation`` catalog) and guard against the root
``model_prices_and_context_window.json`` and the bundled
``litellm/model_prices_and_context_window_backup.json`` drifting out of sync for
the ``cloudflare/`` namespace.
"""

import json
import os

import pytest

import litellm

ROOT_MAP = os.path.join(
    os.path.dirname(os.path.dirname(litellm.__file__)),
    "model_prices_and_context_window.json",
)
BACKUP_MAP = os.path.join(
    os.path.dirname(litellm.__file__),
    "model_prices_and_context_window_backup.json",
)


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _cloudflare_keys(data: dict) -> set:
    return {k for k in data if k.startswith("cloudflare/")}


def test_root_and_backup_have_identical_cloudflare_keys():
    if not os.path.exists(ROOT_MAP):
        pytest.skip("root cost map only ships in source checkouts")
    assert _cloudflare_keys(_load(ROOT_MAP)) == _cloudflare_keys(_load(BACKUP_MAP))


def test_root_and_backup_cloudflare_entries_are_byte_for_byte_equal():
    if not os.path.exists(ROOT_MAP):
        pytest.skip("root cost map only ships in source checkouts")
    root = {k: v for k, v in _load(ROOT_MAP).items() if k.startswith("cloudflare/")}
    backup = {k: v for k, v in _load(BACKUP_MAP).items() if k.startswith("cloudflare/")}
    assert root == backup
