from litellm import get_llm_provider


def test_openai_chat_completions_prefix_is_normalized():
    """Regression for #32489: `openai/chat_completions/<name>` must resolve to the
    OpenAI provider with the bare model id. Without normalization the literal
    `chat_completions/<name>` is forwarded to OpenAI, which rejects it as an
    invalid model id on the /chat/completions path."""
    model, provider, _, _ = get_llm_provider(model="openai/chat_completions/gpt-5.2")

    assert provider == "openai"
    assert model == "gpt-5.2"


def test_openai_chat_completions_prefix_with_nested_path_is_normalized():
    """The remainder after the prefix is preserved verbatim, including further slashes."""
    model, provider, _, _ = get_llm_provider(model="openai/chat_completions/ft:gpt-4o:acme")

    assert provider == "openai"
    assert model == "ft:gpt-4o:acme"


def test_openai_chat_completions_prefix_with_empty_remainder_is_left_untouched():
    """A bare prefix with no model name is not a valid alias and must not be rewritten."""
    model, provider, _, _ = get_llm_provider(model="openai/chat_completions/")

    assert provider == "openai"
    assert model == "chat_completions/"


def test_plain_openai_model_is_unaffected():
    """Models without the alias prefix must resolve exactly as before."""
    model, provider, _, _ = get_llm_provider(model="openai/gpt-4o")

    assert provider == "openai"
    assert model == "gpt-4o"
