"""
NOTE: BETA - this API is currently in development and is subject to change.
"""

import os
import typing as t

from ddtrace import config
from ddtrace.ext.test_visibility._constants import ITR_SKIPPING_LEVEL
from ddtrace.internal.utils.formats import asbool


def _get_default_test_visibility_contrib_config() -> t.Dict[str, t.Any]:
    return dict(
        _default_service="default_test_visibility_service",
        itr_skipping_level=ITR_SKIPPING_LEVEL.SUITE
        if asbool(os.getenv("_DD_CIVISIBILITY_ITR_SUITE_MODE"))
        else ITR_SKIPPING_LEVEL.TEST,
        _itr_skipping_ignore_parameters=False,
    )


# Default test visibility settings
config._add(
    "test_visibility",
    _get_default_test_visibility_contrib_config(),
)


def get_version() -> str:
    return "0.0.1"


__all__ = ["get_version"]
