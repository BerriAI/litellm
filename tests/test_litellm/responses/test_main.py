from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.responses.main import _bridge_kwargs


def test_bridge_kwargs_forward_litellm_owned_kwargs_and_drop_unknown_ones() -> None:
    control = object()
    kwargs = {
        "temperature": 0.2,
        "litellm_trace_id": "trace-1",
        "_litellm_control": control,
        "client_metadata": {"a": "b"},
        "not_a_known_param": 1,
    }

    assert dict(_bridge_kwargs(kwargs, OpenAIResponsesAPIConfig(), None)) == {
        "temperature": 0.2,
        "litellm_trace_id": "trace-1",
        "_litellm_control": control,
    }
