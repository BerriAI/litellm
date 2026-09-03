import sys
from contextlib import contextmanager
from typing import Iterator

import pytest

import litellm.llms as llms_package
from litellm.types.utils import CallTypes

OPENAI_CHAT_TRANSLATION_MODULE = "litellm.llms.openai.chat.guardrail_translation"


@contextmanager
def unimportable(module_path: str) -> Iterator[None]:
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(sys.modules, module_path, None)
        yield


@pytest.fixture(autouse=True)
def reset_guardrail_translation_cache():
    saved = llms_package.endpoint_guardrail_translation_mappings
    llms_package.endpoint_guardrail_translation_mappings = None
    yield
    llms_package.endpoint_guardrail_translation_mappings = saved


def test_discovery_reports_the_handler_package_it_could_not_import():
    with unimportable(OPENAI_CHAT_TRANSLATION_MODULE):
        discovery = llms_package.discover_guardrail_translations()

    assert discovery.unavailable_modules == (OPENAI_CHAT_TRANSLATION_MODULE,)
    assert CallTypes.acompletion not in discovery.mappings
    assert CallTypes.completion not in discovery.mappings
    assert CallTypes.aembedding in discovery.mappings


def test_complete_discovery_reports_nothing_unavailable():
    discovery = llms_package.discover_guardrail_translations()

    assert discovery.unavailable_modules == ()
    assert CallTypes.acompletion in discovery.mappings


def test_an_incomplete_discovery_is_not_cached_for_the_rest_of_the_process():
    with unimportable(OPENAI_CHAT_TRANSLATION_MODULE):
        partial = llms_package.load_guardrail_translation_mappings()

    assert CallTypes.acompletion not in partial
    assert llms_package.endpoint_guardrail_translation_mappings is None

    recovered = llms_package.load_guardrail_translation_mappings()

    assert CallTypes.acompletion in recovered
    assert CallTypes.completion in recovered


def test_a_complete_discovery_is_cached():
    first = llms_package.load_guardrail_translation_mappings()
    second = llms_package.load_guardrail_translation_mappings()

    assert first is second
    assert llms_package.endpoint_guardrail_translation_mappings is first


def test_lookup_recovers_after_a_failed_discovery():
    with unimportable(OPENAI_CHAT_TRANSLATION_MODULE):
        with pytest.raises(ValueError, match="acompletion"):
            llms_package.get_guardrail_translation_mapping(CallTypes.acompletion)

    assert llms_package.get_guardrail_translation_mapping(CallTypes.acompletion) is not None
