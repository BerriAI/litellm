# Unless explicitly stated otherwise all files in this repository are licensed under the BSD-3-Clause License.
# This product includes software developed at Datadog (https://www.datadoghq.com/).
# Copyright 2015-Present Datadog, Inc
# flake8: noqa
"""
Imports for compatibility with Python 2, Python 3 and Google App Engine.
"""
from functools import wraps
import logging
import socket
import sys

# Note: using `sys.version_info` instead of the helper functions defined here
# so that mypy detects version-specific code paths. Currently, mypy doesn't
# support try/except imports for version-specific code paths either.
#
# https://mypy.readthedocs.io/en/stable/common_issues.html#python-version-and-system-platform-checks

# Python 3.x
if sys.version_info[0] >= 3:
    text = str

# Python 2.x
else:
    text = unicode

# Python >= 3.5
if sys.version_info >= (3, 5):
    from inspect import iscoroutinefunction
# Others
else:

    def iscoroutinefunction(*args, **kwargs):
        return False
