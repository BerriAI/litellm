import sys
from typing import Set


ALL_MODULES: Set[str] = set()  # All modules that have been already imported


def get_newly_imported_modules() -> Set[str]:
    global ALL_MODULES
    latest_modules = set(sys.modules.keys())
    new_modules = latest_modules - ALL_MODULES
    ALL_MODULES = latest_modules
    return new_modules
