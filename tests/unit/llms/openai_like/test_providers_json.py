"""A JSON-configured provider is only useful if LiteLLM can resolve its slug.

The failure this guards is the one a user sees: `model="autarcho/<id>"` is an
unknown-provider error until the slug exists in providers.json. Asserting the
base URL and the key env var is what proves the entry does something, rather
than proving a key is present in a file.
"""

from litellm.llms.openai_like.json_loader import JSONProviderRegistry


def test_autarcho_resolves_to_its_own_base_url():
    JSONProviderRegistry.load()

    assert JSONProviderRegistry.exists("autarcho")

    config = JSONProviderRegistry.get("autarcho")
    assert config is not None
    assert config.default_base_url("any-model") == "https://api.autarcho.com/v1"
    assert config.api_key_env == "AUTARCHO_API_KEY"
