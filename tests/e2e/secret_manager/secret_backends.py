"""The secret manager backends the suite can run against, and which one this run uses.

E2E_SECRET_MANAGER both opts the suite in and names the backend, since a proxy runs
one key_management_system and each lane boots its own proxy for it. Adding a
backend is a secret_store_<system>.py module, its entry here,
gateway/secret_manager_<system>_ci_config.yml, and an up_<system> in backend.sh;
the tests and markers stay as they are (test_secret_backends.py checks the set).
"""

from __future__ import annotations

import os
from types import MappingProxyType
from typing import Final

import pytest

from e2e_config import SECRET_MANAGER_OPT_IN_ENV
from secret_store import SecretBackend
from secret_store_cyberark import CYBERARK
from secret_store_hashicorp_vault import HASHICORP_VAULT

BACKENDS: Final = MappingProxyType({backend.system: backend for backend in (HASHICORP_VAULT, CYBERARK)})


def selected_backend() -> SecretBackend:
    system: Final = os.environ.get(SECRET_MANAGER_OPT_IN_ENV, "").strip()
    backend: Final = BACKENDS.get(system)
    if backend is None:
        pytest.fail(
            f"{SECRET_MANAGER_OPT_IN_ENV}={system!r} names no secret manager backend; "
            f"set it to one of {sorted(BACKENDS)}"
        )
    return backend
