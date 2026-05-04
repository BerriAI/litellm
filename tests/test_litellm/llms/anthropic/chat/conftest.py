"""
Local-conftest for ``tests/test_litellm/llms/anthropic/chat``.

Why this exists
---------------
``litellm.model_cost`` is loaded once at ``litellm.__init__`` time. By default
it fetches ``model_prices_and_context_window.json`` from the **main branch on
GitHub** (``litellm.model_cost_map_url``) — *not* from the PR-branch JSON in
the working tree. That works fine in production (operators get new models
without redeploying litellm) but is the wrong default for tests, which need
to validate the code in front of them against the data in front of them.

Several anthropic chat transformation tests (``test_supports_effort_level_*``,
``test_anthropic_model_supports_effort_param_*``) assert per-model flags
like ``supports_max_reasoning_effort`` / ``supports_xhigh_reasoning_effort``
that may exist in the PR-local JSON but not yet in main's JSON. Without this
fixture those tests pass locally (where AGENTS.md tells contributors to set
``LITELLM_LOCAL_MODEL_COST_MAP=True``) but fail in CI (which doesn't set the
env var) — the chicken-and-egg PR adds flag → CI fetches main without flag
→ test fails → flag never lands on main.

The fixture force-loads ``litellm.model_cost`` from the local JSON for every
test in this directory. ``tests/test_litellm/conftest.py`` already snapshots
and restores ``litellm.model_cost`` per-function, so this mutation is safe
and contained.

This is a scoped workaround. The proper fix is to set
``LITELLM_LOCAL_MODEL_COST_MAP=True`` globally in the test workflow once the
~10 inline-set test files have been audited and the few tests that exercise
the remote-fetch / integrity-validation path have been given carve-outs.
Tracked at https://github.com/BerriAI/litellm/issues/27122.
"""

import os

import pytest

import litellm
from litellm.litellm_core_utils.get_model_cost_map import get_model_cost_map


@pytest.fixture(autouse=True)
def _use_pr_local_model_cost_map(monkeypatch):
    """Force ``litellm.model_cost`` to the PR-branch JSON for the duration of
    each test. ``monkeypatch`` reverts the env var after the test; the parent
    conftest restores ``litellm.model_cost`` from its snapshot.
    """
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(
        litellm,
        "model_cost",
        get_model_cost_map(url=litellm.model_cost_map_url),
    )
    yield
