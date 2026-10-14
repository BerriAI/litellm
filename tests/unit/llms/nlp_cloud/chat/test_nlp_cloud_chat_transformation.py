import pytest

from litellm.llms.nlp_cloud.chat.transformation import NLPCloudConfig


@pytest.mark.parametrize(
    ("supplied", "stored"),
    [
        pytest.param({}, {}, id="nothing-supplied"),
        pytest.param(
            {"max_length": 0, "length_no_input": False, "end_sequence": "", "temperature": 0.0},
            {"max_length": 0, "length_no_input": False, "end_sequence": "", "temperature": 0.0},
            id="falsy-values-are-kept",
        ),
        pytest.param({"top_k": 40, "top_p": None, "num_beams": None}, {"top_k": 40}, id="none-is-skipped"),
    ],
)
def test_init_stores_each_supplied_value_except_none_as_class_level_config(
    supplied: dict[str, object], stored: dict[str, object]
) -> None:
    class ScopedNLPCloudConfig(NLPCloudConfig):
        pass

    ScopedNLPCloudConfig(**supplied)

    assert ScopedNLPCloudConfig.get_config() == stored


def test_init_stores_the_supplied_object_itself_rather_than_a_copy() -> None:
    class ScopedNLPCloudConfig(NLPCloudConfig):
        pass

    bad_words = ["darn", "heck"]

    ScopedNLPCloudConfig(bad_words=bad_words)

    assert ScopedNLPCloudConfig.get_config()["bad_words"] is bad_words


def test_init_called_again_with_none_keeps_the_value_stored_earlier() -> None:
    class ScopedNLPCloudConfig(NLPCloudConfig):
        pass

    ScopedNLPCloudConfig(top_k=40)
    ScopedNLPCloudConfig(top_k=None, top_p=0.5)

    assert ScopedNLPCloudConfig.get_config() == {"top_k": 40, "top_p": 0.5}
