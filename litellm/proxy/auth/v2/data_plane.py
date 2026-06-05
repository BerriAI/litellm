import os
from dataclasses import dataclass
from typing import List, Optional

import casbin

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "data_plane.conf")

# Sentinels that mean "any model" in the existing key/team model lists.
_UNRESTRICTED_SENTINELS = {"*", "all-proxy-models", "all-team-models"}


@dataclass
class ModelAccessSubject:
    """Carries the principal's model entitlement as a casbin ABAC attribute.

    The data plane runs on the inference hot path, so access is decided from
    attributes already on the loaded key/team (no per-key policy rows, no policy
    store read). An empty list means unrestricted, matching existing key
    semantics where ``models == []`` allows every model.
    """

    allowed_models: List[str]

    @property
    def unrestricted(self) -> bool:
        if not self.allowed_models:
            return True
        return any(model in _UNRESTRICTED_SENTINELS for model in self.allowed_models)


_enforcer: Optional[casbin.Enforcer] = None


def _get_enforcer() -> casbin.Enforcer:
    global _enforcer
    if _enforcer is None:
        enforcer = casbin.Enforcer(_MODEL_PATH)
        enforcer.add_policy("allow")  # single gate; the matcher does the deciding
        _enforcer = enforcer
    return _enforcer


def can_call_model(allowed_models: Optional[List[str]], requested_model: str) -> bool:
    """Decide whether a principal with ``allowed_models`` may call ``requested_model``."""
    subject = ModelAccessSubject(allowed_models=list(allowed_models or []))
    return _get_enforcer().enforce(subject, requested_model)
