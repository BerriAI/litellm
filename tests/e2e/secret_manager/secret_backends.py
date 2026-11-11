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
