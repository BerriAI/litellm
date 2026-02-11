"""
Local conftest for policy_engine tests.

The parent conftest (tests/test_litellm/conftest.py) inserts the main repo
root into sys.path and imports litellm from there. When running from a
worktree, this means all litellm imports resolve to the main repo's code
instead of the worktree's.

This conftest fixes the path and clears all cached litellm modules so
subsequent imports resolve from the worktree.
"""

import os
import sys

import pytest

# Fix sys.path: insert this worktree's root FIRST and remove the main repo root.
_this_dir = os.path.dirname(os.path.abspath(__file__))
_worktree_root = os.path.abspath(os.path.join(_this_dir, "..", "..", "..", ".."))
sys.path.insert(0, _worktree_root)

# Remove the main repo path that parent conftest inserted
_main_repo = os.path.abspath(os.path.join(_worktree_root, "..", ".."))
sys.path = [p for p in sys.path if os.path.abspath(p) != _main_repo]

# Clear ALL cached litellm modules so they're re-imported from the worktree
_to_remove = [key for key in sys.modules if key == "litellm" or key.startswith("litellm.")]
for key in _to_remove:
    del sys.modules[key]


@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():
    """Override parent conftest - policy engine tests don't need litellm reload."""
    yield
