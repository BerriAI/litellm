import importlib
import sys
from types import ModuleType
from unittest.mock import Mock, patch

import fastapi.dependencies.utils as fastapi_utils

from litellm.proxy.management_endpoints.management_v1 import common


def test_fastapi_compatibility_get_flat_dependant_exists() -> None:
    assert hasattr(common, "get_flat_dependant")
    assert callable(common.get_flat_dependant)


def test_fastapi_compatibility_get_flat_dependant_fallback() -> None:
    fallback_get_dependant = Mock(name="get_dependant")
    fallback_utils = ModuleType("fastapi.dependencies.utils")
    setattr(fallback_utils, "get_dependant", fallback_get_dependant)

    try:
        with patch.dict(sys.modules, {"fastapi.dependencies.utils": fallback_utils}):
            reloaded_common = importlib.reload(common)
            assert reloaded_common.get_flat_dependant is fallback_get_dependant
    finally:
        importlib.reload(common)

    assert common.get_flat_dependant is fastapi_utils.get_flat_dependant
