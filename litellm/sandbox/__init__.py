"""litellm.sandbox: code-interpreter providers (see main.py) plus harness sandboxes.

`sandbox.local(path)` and `sandbox.docker(image, ...)` re-export litellm.harness.sandbox.
They resolve lazily so `import litellm` does not pull in litellm.harness.
"""

import importlib
from typing import Any, Final

_HARNESS_SANDBOX_MODULE: Final = "litellm.harness.sandbox"
_HARNESS_EXPORTS: Final = frozenset(
    {
        "local",
        "docker",
        "LocalSandbox",
        "DockerSandbox",
        "Sandbox",
        "Process",
        "CompletedRun",
    }
)


def __getattr__(name: str) -> Any:
    if name in _HARNESS_EXPORTS:
        return getattr(importlib.import_module(_HARNESS_SANDBOX_MODULE), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
