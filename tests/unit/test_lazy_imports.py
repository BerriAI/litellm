import importlib
import json
from typing import Final

import pytest

import litellm
import litellm.utils
from litellm._lazy_imports_registry import LITELLM_ATTRIBUTES, UTILS_ATTRIBUTES
from tests.test_litellm_rust.support.child_interpreter import run_child_interpreter


def _declared(location: tuple[str, str]) -> object:
    module_path, attr_name = location
    return getattr(importlib.import_module(module_path, package="litellm"), attr_name)


@pytest.mark.parametrize("name", sorted(LITELLM_ATTRIBUTES))
def test_litellm_attribute_is_the_declared_object(name: str):
    assert getattr(litellm, name) is _declared(LITELLM_ATTRIBUTES[name])


@pytest.mark.parametrize("name", sorted(UTILS_ATTRIBUTES))
def test_utils_attribute_is_the_same_object_from_litellm_and_litellm_utils(name: str):
    expected: Final = _declared(UTILS_ATTRIBUTES[name])
    assert getattr(litellm.utils, name) is expected
    assert getattr(litellm, name) is expected


@pytest.mark.parametrize(
    ("instance_name", "class_name"),
    [
        ("in_memory_llm_clients_cache", "LLMClientCache"),
        ("openaiOSeriesConfig", "OpenAIOSeriesConfig"),
        ("openAIGPTConfig", "OpenAIGPTConfig"),
        ("openAIGPTAudioConfig", "OpenAIGPTAudioConfig"),
        ("openAIGPT5Config", "OpenAIGPT5Config"),
        ("nvidiaNimConfig", "NvidiaNimConfig"),
        ("nvidiaNimEmbeddingConfig", "NvidiaNimEmbeddingConfig"),
        ("oci_transformation", "OCIChatConfig"),
        ("priority_reservation_settings", "PriorityReservationSettings"),
        ("logging_callback_manager", "LoggingCallbackManager"),
    ],
)
def test_lazy_instance_is_a_single_instance_of_its_class(instance_name: str, class_name: str):
    instance: Final = getattr(litellm, instance_name)
    assert type(instance) is getattr(litellm, class_name)
    assert getattr(litellm, instance_name) is instance


def test_module_level_http_clients_use_request_timeout(monkeypatch: pytest.MonkeyPatch):
    client_names: Final = ("module_level_client", "module_level_aclient")
    monkeypatch.setattr(litellm, "request_timeout", 7)
    for client_name in client_names:
        monkeypatch.delitem(vars(litellm), client_name, raising=False)
    try:
        assert litellm.module_level_client.timeout == 7
        assert litellm.module_level_aclient.timeout == 7
        assert litellm.module_level_client is litellm.module_level_client
    finally:
        for client_name in client_names:
            vars(litellm).pop(client_name, None)


def test_provider_list_lists_every_provider():
    assert litellm.provider_list == list(litellm.LlmProviders)


@pytest.mark.parametrize(
    "name", ["proxy", "guardrails", "harness", "_service_logger", "anthropic_interface", "messages"]
)
def test_submodule_attribute_is_the_submodule(name: str):
    assert getattr(litellm, name) is importlib.import_module(f"litellm.{name}")


def test_anthropic_attribute_is_the_anthropic_interface():
    assert litellm.anthropic is importlib.import_module("litellm.anthropic_interface")


def test_import_litellm_leaves_oci_and_anthropic_messages_modules_unloaded_until_first_use():
    result: Final = run_child_interpreter(
        "import json, sys, litellm\n"
        "lazy = ('litellm.llms.oci.chat.transformation', 'litellm.messages.dispatch', 'litellm.anthropic_interface')\n"
        "before = [name for name in lazy if name in sys.modules]\n"
        "from litellm import anthropic, anthropic_messages, oci_transformation\n"
        "print(json.dumps([before, [name for name in lazy if name not in sys.modules]]))\n",
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) == [[], []]


@pytest.mark.parametrize("module", [litellm, litellm.utils])
def test_unknown_attribute_raises_attribute_error(module: object):
    with pytest.raises(AttributeError, match="not_a_litellm_attribute"):
        getattr(module, "not_a_litellm_attribute")
    assert not hasattr(module, "not_a_litellm_attribute")


def test_litellm_instances_are_not_attributes_of_litellm_utils():
    assert not hasattr(litellm.utils, "module_level_client")


@pytest.mark.parametrize(
    "module",
    ["litellm.litellm_core_utils.get_litellm_params", "litellm.batches.batch_utils", "litellm.types.utils"],
)
def test_kwargs_funnel_and_its_importers_load_first_in_fresh_process(module: str):
    """These modules are often the first to pull in litellm.types.utils, and the WIF key sets shared between
    the funnel and all_litellm_params must not turn that into a cycle."""
    result = run_child_interpreter(f"import {module}", timeout=120)
    assert result.returncode == 0, result.stderr
