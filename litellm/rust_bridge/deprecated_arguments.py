import inspect
import os
import sys
import traceback
import warnings
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Final

_LITELLM_ROOT: Final = f"{Path(__file__).parents[1]}{os.sep}"

_DEPRECATED: Final = MappingProxyType(
    {
        "client": (
            "client= is deprecated: litellm is moving requests to Rust, which cannot use a caller's HTTP client. "
            "Set network options with litellm.ssl_verify, litellm.ssl_certificate, or proxy and SSL environment "
            "variables instead"
        )
    }
)


def _first_stacklevel_outside_litellm() -> int:
    return next(
        (
            level
            for level, (frame, _) in enumerate(traceback.walk_stack(inspect.currentframe()))
            if not frame.f_code.co_filename.startswith(_LITELLM_ROOT)
        ),
        1,
    )


def warn_deprecated(name: str) -> None:
    if sys.version_info >= (3, 12):
        warnings.warn(_DEPRECATED[name], DeprecationWarning, skip_file_prefixes=(_LITELLM_ROOT,))
        return
    warnings.warn(_DEPRECATED[name], DeprecationWarning, stacklevel=_first_stacklevel_outside_litellm())


def warn_deprecated_in(kwargs: Mapping[str, object]) -> None:
    for name in (name for name in _DEPRECATED if kwargs.get(name) is not None):
        warn_deprecated(name)
