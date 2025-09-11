import json
import platform
import sys
from typing import Dict

from . import __version__ as pyjwt_version

try:
    import cryptography

    cryptography_version = cryptography.__version__
except ModuleNotFoundError:
    cryptography_version = ""


def info() -> Dict[str, Dict[str, str]]:
    """
    Generate information for a bug report.
    Based on the requests package help utility module.
    """
    try:
        platform_info = {
            "system": platform.system(),
            "release": platform.release(),
        }
    except OSError:
        platform_info = {"system": "Unknown", "release": "Unknown"}

    implementation = platform.python_implementation()

    if implementation == "CPython":
        implementation_version = platform.python_version()
    elif implementation == "PyPy":
        pypy_version_info = sys.pypy_version_info  # type: ignore[attr-defined]
        implementation_version = (
            f"{pypy_version_info.major}."
            f"{pypy_version_info.minor}."
            f"{pypy_version_info.micro}"
        )
        if pypy_version_info.releaselevel != "final":
            implementation_version = "".join(
                [implementation_version, pypy_version_info.releaselevel]
            )
    else:
        implementation_version = "Unknown"

    return {
        "platform": platform_info,
        "implementation": {
            "name": implementation,
            "version": implementation_version,
        },
        "cryptography": {"version": cryptography_version},
        "pyjwt": {"version": pyjwt_version},
    }


def main() -> None:
    """Pretty-print the bug information as JSON."""
    print(json.dumps(info(), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
