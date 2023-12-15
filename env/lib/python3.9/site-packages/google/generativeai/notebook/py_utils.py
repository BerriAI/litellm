# -*- coding: utf-8 -*-
# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convenience functions for writing to and reading from Python variables."""
from __future__ import annotations

import builtins
import keyword
import sys
from typing import Any


def validate_var_name(var_name: str) -> None:
    """Validates that the variable name is a valid identifier."""
    if not var_name.isidentifier():
        raise ValueError('Invalid Python variable name, got "{}"'.format(var_name))
    if keyword.iskeyword(var_name):
        raise ValueError('Cannot use Python keywords, got "{}"'.format(var_name))


def get_main_module():
    return sys.modules["__main__"]


def get_py_var(var_name: str) -> Any:
    """Retrieves the value of `var_name` from the global environment."""
    validate_var_name(var_name)
    g_vars = vars(get_main_module())
    if var_name in g_vars:
        return g_vars[var_name]
    elif var_name in vars(builtins):
        return vars(builtins)[var_name]
    raise NameError('"{}" not found'.format(var_name))


def has_py_var(var_name: str) -> bool:
    """Returns true if `var_name` is defined in the global environment."""
    try:
        validate_var_name(var_name)
        _ = get_py_var(var_name)
    except ValueError:
        return False
    except NameError:
        return False

    return True


def set_py_var(var_name: str, val: Any) -> None:
    """Sets the value of `var_name` in the global environment."""
    validate_var_name(var_name)
    g_vars = vars(get_main_module())
    g_vars[var_name] = val
