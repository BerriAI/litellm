"""The shared chat transformation config contract."""

from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig


def test_merge_extra_body_is_a_shallow_merge_by_default():
    cfg = OpenAIGPTConfig()
    request = {"model": "gpt-5.5", "metadata": {"completion_window": "flex", "trace_id": "a"}}
    extra_body = {"metadata": {"trace_id": "b"}, "reasoning_budget": 128}

    merged = cfg.merge_extra_body(request, extra_body)

    assert merged == {
        "model": "gpt-5.5",
        "metadata": {"trace_id": "b"},
        "reasoning_budget": 128,
    }
    assert request == {"model": "gpt-5.5", "metadata": {"completion_window": "flex", "trace_id": "a"}}
    assert cfg.merge_extra_body(request, None) == request
