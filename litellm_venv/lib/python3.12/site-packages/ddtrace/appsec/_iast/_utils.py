from functools import lru_cache
import sys
from typing import List

from ddtrace.internal.logger import get_logger
from ddtrace.settings.asm import config as asm_config


@lru_cache(maxsize=1)
def _is_python_version_supported() -> bool:
    # IAST supports Python versions 3.6 to 3.13
    return (3, 6, 0) <= sys.version_info < (3, 14, 0)


def _is_iast_enabled():
    if not asm_config._iast_enabled:
        return False

    if not _is_python_version_supported():
        log = get_logger(__name__)
        log.info("IAST is not compatible with the current Python version")
        return False

    return True


def _get_source_index(sources: List, source) -> int:
    i = 0
    for source_ in sources:
        if hash(source_) == hash(source):
            return i
        i += 1
    return -1


def _is_iast_debug_enabled():
    return asm_config._iast_debug


def _is_iast_propagation_debug_enabled():
    return asm_config._iast_propagation_debug
