from litellm.llms.deprecated_providers.palm import PalmConfig


def test_default_construction_contributes_no_request_params():
    assert PalmConfig().get_config() == {}


def test_constructor_arguments_become_config_of_the_constructed_class_only():
    class ScopedPalmConfig(PalmConfig):
        pass

    ScopedPalmConfig(temperature=0.0, top_k=40, examples=[])

    assert ScopedPalmConfig.get_config() == {"temperature": 0.0, "top_k": 40, "examples": []}
    assert PalmConfig.get_config() == {}
