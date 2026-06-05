"""Backwards-compatible shim. Auth logic now lives in :mod:`litellm.auth`.

Importing from ``litellm.proxy.auth`` continues to work so external consumers
(including the published ``litellm-enterprise`` package and user-defined custom
auth hooks) are not broken, but new code should import from ``litellm.auth``.
The submodules are aliased to the exact module objects loaded under
``litellm.auth`` so class identity is preserved (no double import).
"""

import importlib
import pkgutil
import sys
import warnings

import litellm.auth as _new_pkg

warnings.warn(
    "litellm.proxy.auth has moved to litellm.auth; update your imports. "
    "The old path will keep working as a compatibility shim",
    DeprecationWarning,
    stacklevel=2,
)

for _submodule in pkgutil.iter_modules(_new_pkg.__path__):
    _old_name = f"{__name__}.{_submodule.name}"
    if _old_name not in sys.modules:
        sys.modules[_old_name] = importlib.import_module(
            f"litellm.auth.{_submodule.name}"
        )

sys.modules[__name__] = _new_pkg
