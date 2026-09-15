from typing import Final

from litellm.rust_bridge.params import provider_param_names


def test_selects_unknown_fields_and_consumed_controls_without_reading_values() -> None:
    callback: Final = object()
    future: Final = {"nested": [None, False, 0]}
    kwargs: Final = {
        "model": "mistral/model",
        "litellm_logging_obj": callback,
        "metadata": callback,
        "vertex_credentials": "credentials",
        "future": future,
        "extra_body": {"future": None},
    }

    assert provider_param_names(kwargs, ("vertex_credentials",)) == ("vertex_credentials", "future", "extra_body")
    assert kwargs["future"] is future
    assert kwargs["litellm_logging_obj"] is callback


def test_route_can_explicitly_consume_a_name_also_used_by_sdk() -> None:
    assert provider_param_names({"metadata": {"provider": True}}, ("metadata",)) == ("metadata",)
