"""
Unit tests for litellm/llms/base_llm/batches/transformation.py

BaseBatchesConfig is the abstract base class that every provider-specific
batches config subclasses. It is almost entirely interface (abstractmethods +
one abstract property), so the only concrete behavior to regression-lock is:

  - the abstractness contract: the base class cannot be instantiated, and a
    subclass missing any abstract member also cannot be instantiated; a
    subclass implementing all of them can.
  - get_config(): a classmethod that reflects over ``cls.__dict__`` and returns
    the class-level config attributes, filtering out dunders, ``_abc`` internals,
    callables (function/builtin/classmethod/staticmethod), and ``None`` values.

These tests assert the exact dict get_config() produces for hand-built
subclasses, so a change to the filter predicate (e.g. dropping the ``None``
filter, dropping the staticmethod/classmethod filter, or widening the prefix
filter to all single-underscore names) makes a test fail.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.base_llm.batches.transformation import BaseBatchesConfig
from litellm.types.utils import LlmProviders


# --------------------------------------------------------------------------- #
# A fully-concrete subclass: implements every abstract member with trivial
# bodies so it can be instantiated and so get_config() has a real cls to
# reflect over. Class-level attributes here are the get_config() fixtures.
# --------------------------------------------------------------------------- #


class _ConcreteBatchesConfig(BaseBatchesConfig):
    string_attr = "hello"
    int_attr = 42
    list_attr = [1, 2, 3]
    none_attr = None
    _single_underscore = "kept"

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.OPENAI

    def validate_environment(
        self,
        headers,
        model,
        messages,
        optional_params,
        litellm_params,
        api_key=None,
        api_base=None,
    ) -> dict:
        return headers

    def get_complete_batch_url(
        self, api_base, api_key, model, optional_params, litellm_params, data
    ) -> str:
        return "https://example.com/batch"

    def transform_create_batch_request(
        self, model, create_batch_data, optional_params, litellm_params
    ):
        return {"created": True}

    def transform_create_batch_response(
        self, model, raw_response, logging_obj, litellm_params
    ):
        return raw_response

    def transform_retrieve_batch_request(
        self, batch_id, optional_params, litellm_params
    ):
        return {"batch_id": batch_id}

    def transform_retrieve_batch_response(
        self, model, raw_response, logging_obj, litellm_params
    ):
        return raw_response

    def get_error_class(self, error_message, status_code, headers):
        return Exception(error_message)


# =========================================================================== #
# Abstractness contract
# =========================================================================== #


def test_base_class_cannot_be_instantiated():
    """The base class has unimplemented abstractmethods, so direct
    instantiation must raise TypeError."""
    with pytest.raises(TypeError):
        BaseBatchesConfig()


def test_fully_concrete_subclass_can_be_instantiated():
    instance = _ConcreteBatchesConfig()
    assert isinstance(instance, BaseBatchesConfig)


@pytest.mark.parametrize(
    "missing_member",
    [
        "custom_llm_provider",
        "validate_environment",
        "get_complete_batch_url",
        "transform_create_batch_request",
        "transform_create_batch_response",
        "transform_retrieve_batch_request",
        "transform_retrieve_batch_response",
        "get_error_class",
    ],
)
def test_subclass_missing_any_abstract_member_cannot_instantiate(missing_member):
    """Every abstract member is part of the contract: dropping any one of them
    leaves the subclass abstract and uninstantiable."""
    namespace = {
        k: v
        for k, v in _ConcreteBatchesConfig.__dict__.items()
        if not k.startswith("__")
    }
    namespace.pop(missing_member)
    Incomplete = type("Incomplete", (BaseBatchesConfig,), namespace)
    with pytest.raises(TypeError):
        Incomplete()


def test_concrete_instance_methods_run():
    """Sanity: the trivial overrides actually execute through the base contract."""
    instance = _ConcreteBatchesConfig()
    assert instance.custom_llm_provider == LlmProviders.OPENAI
    assert instance.validate_environment(
        headers={"x": "1"},
        model="m",
        messages=[],
        optional_params={},
        litellm_params={},
    ) == {"x": "1"}
    assert instance.transform_retrieve_batch_request(
        batch_id="b-1", optional_params={}, litellm_params={}
    ) == {"batch_id": "b-1"}


# =========================================================================== #
# get_config()
# =========================================================================== #


def test_get_config_returns_class_level_non_none_data_attrs():
    """Exact contents: only class-level data attributes that are not None,
    not dunders, not callables. Single-underscore names ARE kept (only ``__``
    and ``_abc`` prefixes are filtered). The ``custom_llm_provider`` property
    object also survives the filter (a property is neither a function nor None),
    matching how real provider subclasses define it."""
    config = _ConcreteBatchesConfig.get_config()
    custom_llm_provider = config.pop("custom_llm_provider")
    assert isinstance(custom_llm_provider, property)
    assert config == {
        "string_attr": "hello",
        "int_attr": 42,
        "list_attr": [1, 2, 3],
        "_single_underscore": "kept",
    }


def test_get_config_excludes_none_valued_attrs():
    assert "none_attr" not in _ConcreteBatchesConfig.get_config()


def test_get_config_excludes_methods_and_property():
    config = _ConcreteBatchesConfig.get_config()
    for method_name in (
        "validate_environment",
        "get_complete_batch_url",
        "transform_create_batch_request",
        "transform_create_batch_response",
        "transform_retrieve_batch_request",
        "transform_retrieve_batch_response",
        "get_error_class",
        "get_config",
    ):
        assert method_name not in config


def test_get_config_excludes_classmethod_and_staticmethod():
    """classmethod and staticmethod objects are filtered even though they are
    not plain FunctionType."""

    class WithCallables(_ConcreteBatchesConfig):
        keep_me = "yes"

        @staticmethod
        def a_static():
            return 1

        @classmethod
        def a_class(cls):
            return 2

    config = WithCallables.get_config()
    assert config == {"keep_me": "yes"}


def test_get_config_only_reflects_own_dict_not_inherited():
    """get_config reflects cls.__dict__ only, so attributes defined on a parent
    do not leak into a child's config."""

    class Parent(_ConcreteBatchesConfig):
        parent_attr = "parent"

    class Child(Parent):
        child_attr = "child"

    assert Parent.get_config() == {"parent_attr": "parent"}
    assert Child.get_config() == {"child_attr": "child"}


def test_get_config_on_base_class_exposes_only_the_abstract_property():
    """On the base class itself, the only ``__dict__`` member that survives the
    filter is the ``custom_llm_provider`` property object (a property is neither
    a function nor None and its name has no filtered prefix)."""
    config = BaseBatchesConfig.get_config()
    assert list(config.keys()) == ["custom_llm_provider"]
    assert isinstance(config["custom_llm_provider"], property)
