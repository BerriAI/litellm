import pytest

from litellm.rust_bridge.host.provider import resolved_provider


@pytest.mark.parametrize(
    ("model", "custom_llm_provider", "expected"),
    [
        ("vertex_ai/claude-opus-4-8", None, ("claude-opus-4-8", "vertex_ai")),
        ("claude-opus-4-8", "bedrock", ("claude-opus-4-8", "bedrock")),
        ("totally-unknown-model", None, ("totally-unknown-model", "anthropic")),
    ],
    ids=["provider_prefix_is_split_off", "explicit_provider_wins", "unroutable_model_falls_back_to_anthropic"],
)
def test_resolved_provider(model: str, custom_llm_provider: str | None, expected: tuple[str, str]) -> None:
    assert resolved_provider(model, custom_llm_provider) == expected
