from litellm import MorphChatConfig


def test_morph_supported_params():
    config = MorphChatConfig()
    apply_params = config.get_supported_openai_params("morph/morph-v3-large")
    chat_params = config.get_supported_openai_params("morph/morph-kimik3")

    assert apply_params == ["messages", "model", "stream", "temperature", "stop", "max_tokens"]
    assert set(chat_params) == {
        "messages",
        "model",
        "stream",
        "temperature",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
        "stop",
        "seed",
        "max_tokens",
        "logit_bias",
        "tools",
        "response_format",
        "logprobs",
    }


def test_morph_maps_supported_chat_params():
    config = MorphChatConfig()
    non_default_params = {
        "max_completion_tokens": 1024,
        "response_format": {"type": "json_object"},
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
    }

    mapped_params = config.map_openai_params(
        non_default_params=non_default_params,
        optional_params={},
        model="morph-kimik3",
        drop_params=False,
    )

    assert mapped_params == {
        "max_tokens": 1024,
        "response_format": {"type": "json_object"},
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
    }
