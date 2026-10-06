from litellm.llms.nvidia_nim.embed import NvidiaNimEmbeddingConfig


def test_map_openai_params_routes_nvidia_params_and_kwargs_into_extra_body():
    mapped = NvidiaNimEmbeddingConfig().map_openai_params(
        non_default_params={"input_type": "query", "truncate": "END", "dimensions": 512},
        optional_params={},
        kwargs={"modality": "text"},
    )

    assert mapped == {
        "extra_body": {"input_type": "query", "truncate": "END", "modality": "text"},
        "dimensions": 512,
    }


def test_map_openai_params_extends_an_existing_extra_body():
    mapped = NvidiaNimEmbeddingConfig().map_openai_params(
        non_default_params={"truncate": "NONE"},
        optional_params={"extra_body": {"input_type": "passage"}, "user": "user-1"},
    )

    assert mapped == {"extra_body": {"input_type": "passage", "truncate": "NONE"}, "user": "user-1"}


def test_constructor_arguments_become_config_of_the_constructed_class_only():
    class ScopedNvidiaNimEmbeddingConfig(NvidiaNimEmbeddingConfig):
        pass

    ScopedNvidiaNimEmbeddingConfig(input_type="passage", truncate="END")

    assert ScopedNvidiaNimEmbeddingConfig.get_config() == {"input_type": "passage", "truncate": "END"}
    assert NvidiaNimEmbeddingConfig.get_config() == {}
