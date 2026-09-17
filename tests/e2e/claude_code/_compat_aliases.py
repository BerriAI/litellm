"""Per-test copies of the compat deployments.

The compat cells probe shared virtual names like
``claude-sonnet-4-5-bedrock-invoke``. One deployment per name is registered for
the whole session, which leaves it with no owning test, so its provider calls
cannot be attributed and are forwarded live. Registering a second deployment
under the same name would not help: the router would have two candidates and
load-balance between them, so a cell would keep landing on another cell's
deployment.

A cell therefore gets its own *name*. The suffix is a digest of the node id, so
it is unique per cell and identical from one build to the next, and the
deployment is registered from inside the test, which is what gives it the
test's ownership segment and makes its calls cacheable.
"""

from __future__ import annotations

import threading
from typing import Callable, Dict, Final, Optional, Protocol

from fixture_bundle import slug_for_test

from claude_code._compat_models import CompatDeployment


class Registrar(Protocol):
    def create_model(self, model_name: str, litellm_params: object) -> str: ...

    def delete_model(self, model_id: str) -> None: ...


class AliasRegistry:
    """One test's private copies, registered on first use and deleted after."""

    def __init__(
        self, registrar: Registrar, deployments: Dict[str, CompatDeployment], test_key: str
    ) -> None:
        self._registrar = registrar
        self._deployments = deployments
        self._suffix = slug_for_test(test_key)
        self._aliases: Dict[str, str] = {}
        self._model_ids: Dict[str, str] = {}
        self._lock = threading.Lock()

    def alias(self, model_name: str) -> str:
        """The name this test should send for `model_name`.

        Unknown names pass through untouched: only the compat matrix's own
        deployments are re-registered here, and a cell probing anything else is
        exercising a deployment somebody else owns."""
        deployment = self._deployments.get(model_name)
        if deployment is None:
            return model_name
        with self._lock:
            existing = self._aliases.get(model_name)
            if existing is not None:
                return existing
            alias = f"{model_name}--{self._suffix}"
            model_id = self._registrar.create_model(alias, deployment.litellm_params)
            self._aliases[model_name] = alias
            self._model_ids[alias] = model_id
            return alias

    def teardown(self) -> None:
        for model_id in self._model_ids.values():
            try:
                self._registrar.delete_model(model_id)
            except Exception:  # noqa: BLE001  # teardown must not mask a test's own failure
                pass


_ACTIVE: Optional[AliasRegistry] = None
_ACTIVE_LOCK: Final = threading.Lock()


def install(registry: Optional[AliasRegistry]) -> None:
    global _ACTIVE  # rebind-ok: one process-wide slot the per-test fixture owns
    with _ACTIVE_LOCK:
        _ACTIVE = registry


def resolve(model_name: str) -> str:
    """The name to send for `model_name`, unchanged when no registry is installed."""
    registry: Final = _ACTIVE
    return model_name if registry is None else registry.alias(model_name)


Resolver = Callable[[str], str]
