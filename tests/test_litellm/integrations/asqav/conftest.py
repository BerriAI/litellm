# Minimal conftest for the asqav standalone tests.
# These tests do not import litellm at module scope and do not need the
# parent conftest fixtures (which pull in the full litellm stack).
import os
import sys

# Ensure the repo root is on sys.path so "litellm.integrations.asqav" resolves.
_REPO_ROOT = os.path.abspath(
    os.path.join(__file__, "..", "..", "..", "..", "..", "..")
)
sys.path.insert(0, _REPO_ROOT)
