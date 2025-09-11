#!/usr/bin/env python3

from sys import version_info


if version_info < (3, 7, 0):
    from .module_names_py36 import STDLIB_MODULE_NAMES
elif version_info < (3, 8, 0):
    from .module_names_py37 import STDLIB_MODULE_NAMES
elif version_info < (3, 9, 0):
    from .module_names_py38 import STDLIB_MODULE_NAMES
elif version_info < (3, 10, 0):
    from .module_names_py39 import STDLIB_MODULE_NAMES
elif version_info < (3, 11, 0):
    from .module_names_py310 import STDLIB_MODULE_NAMES
elif version_info < (3, 12, 0):
    from .module_names_py311 import STDLIB_MODULE_NAMES
else:
    from .module_names_py312 import STDLIB_MODULE_NAMES


def _stdlib_for_python_version():  # type: () -> set
    return STDLIB_MODULE_NAMES
