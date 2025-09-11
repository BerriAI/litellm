import sys

from ddtrace.profiling import _build

from .profiler import Profiler  # noqa:F401


def _not_compatible_abi():
    raise ImportError(
        "Python ABI is not compatible, you need to recompile this module.\n"
        "Reinstall it with the following command:\n"
        "  pip install --no-binary ddtrace ddtrace"
    )


if (3, 7) < _build.compiled_with <= (3, 7, 3):
    if sys.version_info[:3] > (3, 7, 3):
        _not_compatible_abi()
elif (3, 7, 3) < _build.compiled_with < (3, 8):
    if (3, 7) < sys.version_info[:3] <= (3, 7, 3):
        _not_compatible_abi()
