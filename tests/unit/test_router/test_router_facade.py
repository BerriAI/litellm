"""
The ``litellm.Router`` facade's own helpers in litellm/router.py: how it binds the constructor
arguments and which members it leaves to the facade.
"""

import pytest

from litellm.router import Router, _class_descriptor, router_arguments


def test_router_arguments_binds_positional_arguments_to_their_parameter_names():
    model_list = [{"model_name": "g", "litellm_params": {"model": "openai/gpt-4o-mini"}}]

    arguments = router_arguments(model_list, num_retries=2)

    assert dict(arguments) == {"model_list": model_list, "num_retries": 2}


def test_router_arguments_leaves_unset_parameters_out():
    assert dict(router_arguments()) == {}


def test_router_arguments_rejects_an_unknown_argument_naming_the_router():
    with pytest.raises(TypeError, match=r"^Router\.__init__\(\) got an unexpected keyword argument 'not_a_setting'"):
        router_arguments(not_a_setting=1)


def test_router_arguments_are_read_only():
    arguments = router_arguments(num_retries=2)

    with pytest.raises(TypeError):
        arguments["num_retries"] = 3  # pyright: ignore[reportIndexIssue]  # proving the mapping is read-only


def test_class_descriptor_is_true_for_a_data_descriptor_on_the_class():
    assert _class_descriptor(Router, "_core")


def test_class_descriptor_is_false_for_a_plain_method_or_a_missing_name():
    assert not _class_descriptor(Router, "from_config")
    assert not _class_descriptor(Router, "no_such_member")
