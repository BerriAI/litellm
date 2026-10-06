from litellm.constants import DEFAULT_MAX_TOKENS
from litellm.llms.deprecated_providers.aleph_alpha import AlephAlphaConfig


def test_default_construction_leaves_only_the_required_maximum_tokens_in_config():
    assert AlephAlphaConfig().get_config() == {"maximum_tokens": DEFAULT_MAX_TOKENS}


def test_constructor_arguments_become_config_of_the_constructed_class_only():
    class ScopedAlephAlphaConfig(AlephAlphaConfig):
        pass

    ScopedAlephAlphaConfig(maximum_tokens=64, echo=False, stop_sequences=["END"])

    assert ScopedAlephAlphaConfig.get_config() == {"maximum_tokens": 64, "echo": False, "stop_sequences": ["END"]}
    assert AlephAlphaConfig.get_config() == {"maximum_tokens": DEFAULT_MAX_TOKENS}
