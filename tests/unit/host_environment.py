import os
from typing import Final

WINDOWS_HOST_ENVIRONMENT: Final = frozenset(
    (
        "SYSTEMROOT",
        "SYSTEMDRIVE",
        "WINDIR",
        "COMSPEC",
        "PATHEXT",
        "USERPROFILE",
        "HOMEDRIVE",
        "HOMEPATH",
        "APPDATA",
        "LOCALAPPDATA",
    )
)
HOST_ENVIRONMENT_ALLOWLIST: Final = frozenset(
    (
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "TMPDIR",
        "TEMP",
        "TMP",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TZ",
        "VIRTUAL_ENV",
        "LITELLM_LOCAL_MODEL_COST_MAP",
        "TIKTOKEN_CACHE_DIR",
    )
) | (WINDOWS_HOST_ENVIRONMENT if os.name == "nt" else frozenset())
HOST_ENVIRONMENT_ALLOWED_PREFIXES: Final = ("PYTEST_", "PYTHON", "COV_CORE_", "COVERAGE_")


def is_host_only(name: str) -> bool:
    return name not in HOST_ENVIRONMENT_ALLOWLIST and not name.startswith(HOST_ENVIRONMENT_ALLOWED_PREFIXES)
