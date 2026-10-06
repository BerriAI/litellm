import pytest

from litellm.llms.volcengine.embedding.transformation import VolcEngineEmbeddingConfig

MODEL = "doubao-embedding-text-240715"


@pytest.mark.parametrize(
    ("embedding_input", "optional_params", "expected"),
    [
        ("hello", {}, {"model": MODEL, "input": ["hello"]}),
        (
            ["hello", "world"],
            {"encoding_format": "base64", "user": "user-1"},
            {"model": MODEL, "input": ["hello", "world"], "encoding_format": "base64", "user": "user-1"},
        ),
        ("hello", {"encoding_format": None, "user": None}, {"model": MODEL, "input": ["hello"]}),
    ],
)
def test_transform_embedding_request_builds_the_volcengine_body(
    embedding_input: str | list[str], optional_params: dict[str, object], expected: dict[str, object]
):
    body = VolcEngineEmbeddingConfig().transform_embedding_request(
        model=MODEL, input=embedding_input, optional_params=optional_params, headers={}
    )

    assert body == expected


def test_map_openai_params_keeps_supported_params():
    mapped = VolcEngineEmbeddingConfig().map_openai_params(
        non_default_params={"encoding_format": "base64", "user": "user-1", "extra_headers": {"x-trace": "1"}},
        optional_params={},
        model=MODEL,
        drop_params=False,
    )

    assert mapped == {"encoding_format": "base64", "user": "user-1", "extra_headers": {"x-trace": "1"}}


@pytest.mark.parametrize(
    ("non_default_params", "message"),
    [
        ({"encoding_format": "int8"}, "Unsupported encoding_format: int8"),
        ({"dimensions": 256}, "Unsupported parameter for Volcengine: dimensions"),
    ],
)
def test_map_openai_params_rejects_unsupported_params(non_default_params: dict[str, object], message: str):
    with pytest.raises(ValueError, match=message):
        VolcEngineEmbeddingConfig().map_openai_params(
            non_default_params=non_default_params, optional_params={}, model=MODEL, drop_params=False
        )


@pytest.mark.parametrize("non_default_params", [{"encoding_format": "int8"}, {"dimensions": 256}])
def test_map_openai_params_drops_unsupported_params_when_drop_params_is_set(non_default_params: dict[str, object]):
    mapped = VolcEngineEmbeddingConfig().map_openai_params(
        non_default_params=non_default_params, optional_params={}, model=MODEL, drop_params=True
    )

    assert mapped == {}


def test_constructor_arguments_become_config_of_the_constructed_class_only():
    class ScopedVolcEngineEmbeddingConfig(VolcEngineEmbeddingConfig):
        pass

    ScopedVolcEngineEmbeddingConfig(encoding_format="base64")

    assert ScopedVolcEngineEmbeddingConfig.get_config() == {"encoding_format": "base64"}
    assert VolcEngineEmbeddingConfig.get_config() == {}
