from typing import Final

from integration._support.client import Gateway, list_value, object_value, string_value

PASS_THROUGH_PROVIDERS: Final = frozenset(
    {"openrouter", "litellm_proxy", "hosted_vllm", "custom_openai", "vllm", "huggingface"}
)


def _chat_completion_providers(gateway: Gateway) -> frozenset[str]:
    entries: Final = list_value(gateway.get("/public/endpoints")["endpoints"])
    entry: Final = next(object_value(value) for value in entries if object_value(value)["key"] == "chat_completions")
    return frozenset(string_value(object_value(provider)["slug"]) for provider in list_value(entry["providers"]))


def test_pass_through_providers_are_listed_under_chat_completions(gateway: Gateway) -> None:
    listed: Final = _chat_completion_providers(gateway)
    missing: Final = sorted(PASS_THROUGH_PROVIDERS - listed)

    assert missing == [], f"pass-through providers missing from chat_completions on /public/endpoints: {missing}"
