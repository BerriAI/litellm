import os
import sys
import typing
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

if typing.TYPE_CHECKING:
    from litellm import _lazy_module
else:
    from types import ModuleType

    _lazy_module = ModuleType


@contextmanager
def context_env(**kwargs):
    """
    Context manager to temporarily set environment variables.
    """
    old_env = {k: os.environ.get(k) for k in kwargs}
    os.environ.update(kwargs)
    try:
        yield
    finally:
        for k, v in old_env.items():
            if v is None:
                del os.environ[k]
            else:
                os.environ[k] = v


def import_lazy_module() -> _lazy_module:

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "litellm._lazy_module",
        str((Path(__file__).parents[2] / "litellm/_lazy_module.py")),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_import_force_static_import():
    with context_env(LITELLM_LAZY_IMPORT="False"):
        import litellm


def test_import_lazy_import():
    import litellm

    print(litellm.Choices)
    print(litellm.default_redis_batch_cache_expiry)


def test_lazy_module_default_behavior():
    # Dynamic import of litellm module
    _lazy_module = import_lazy_module()

    mod = _lazy_module._LazyModule(
        "litellm",
        __file__,
        {"litellm.llms.base": ["BaseLLM"]},
        module_spec=__spec__,
    )
    with context_env(LITELLM_LAZY_IMPORT="False"):
        from litellm.llms.base import BaseLLM

        assert mod.BaseLLM is BaseLLM


def test_lazy_module_alias():
    # Dynamic import of litellm module
    _lazy_module = import_lazy_module()

    mod = _lazy_module._LazyModule(
        "litellm",
        __file__,
        {"litellm.llms.base": [_lazy_module._Alias("BaseLLM", "LLM")]},
        module_spec=__spec__,
    )
    with context_env(LITELLM_LAZY_IMPORT="False"):
        from litellm.llms.base import BaseLLM

        assert mod.LLM is BaseLLM


def test_import_with_no_exception():
    # No exception should be raised
    from litellm import proxy
    from litellm.llms import base


def test_lazy_import_with_patch():
    import litellm

    with patch("litellm.api_base", "https://litellm-api-base.example.com/v1"):

        assert litellm.api_base == "https://litellm-api-base.example.com/v1"


def test_dir_lazy_module():
    # Dynamic import of litellm module
    _lazy_module = import_lazy_module()

    mod = _lazy_module._LazyModule(
        "litellm",
        __file__,
        {"litellm.llms.base": ["BaseLLM"]},
        module_spec=__spec__,
    )
    assert "BaseLLM" in dir(mod)


def test_has_extra_object():
    # Dynamic import of litellm module
    _lazy_module = import_lazy_module()

    mod = _lazy_module._LazyModule(
        "litellm",
        __file__,
        {"litellm.llms.base": ["BaseLLM"]},
        module_spec=__spec__,
        extra_object={"extra": "object"},
    )
    assert hasattr(mod, "extra")
    assert mod.extra == "object"


def test_comparing_lazy_module_with_static_import():
    # Dynamic import of litellm module
    _lazy_module = import_lazy_module()

    mod = _lazy_module._LazyModule(
        "litellm",
        __file__,
        {"litellm.llms.base": ["BaseLLM"]},
        module_spec=__spec__,
    )
    import importlib

    with context_env(LITELLM_LAZY_IMPORT="False"):
        del sys.modules["litellm"]
        import litellm

        litellm_static = importlib.reload(litellm)
        litellm_static = litellm

    litellm_dynamic = importlib.reload(litellm)

    assert litellm_dynamic is not litellm_static

    fields_litellm_static = frozenset(
        [name for name in dir(litellm_static) if name.startswith("litellm")]
    )
    fields_litellm_dynamic = frozenset(dir(litellm_dynamic))

    # Get the difference between the two sets
    difference = fields_litellm_static - fields_litellm_dynamic
    assert len(difference) == 0, f"Difference: {difference}"

    # Check obtining the same object
    for field in fields_litellm_static:
        getattr(litellm_dynamic, field)  # Expect no error
