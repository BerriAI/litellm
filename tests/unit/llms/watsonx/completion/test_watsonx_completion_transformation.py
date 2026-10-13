import pytest

from litellm.llms.watsonx.completion.transformation import IBMWatsonXAIConfig


@pytest.mark.parametrize(
    ("supplied", "stored"),
    [
        pytest.param({}, {"kwargs": {}}, id="nothing-supplied"),
        pytest.param(
            {"decoding_method": "", "max_new_tokens": 0, "include_stop_sequences": False, "stream": False},
            {
                "decoding_method": "",
                "max_new_tokens": 0,
                "include_stop_sequences": False,
                "stream": False,
                "kwargs": {},
            },
            id="falsy-values-are-kept",
        ),
        pytest.param(
            {"top_k": 40, "top_p": None, "random_seed": None}, {"top_k": 40, "kwargs": {}}, id="none-is-skipped"
        ),
        pytest.param(
            {"temperature": 0.2, "project_id": "proj-1", "space_id": None},
            {"temperature": 0.2, "kwargs": {"project_id": "proj-1", "space_id": None}},
            id="unknown-keywords-are-grouped-under-kwargs",
        ),
    ],
)
def test_init_stores_each_supplied_value_except_none_as_class_level_config(
    supplied: dict[str, object], stored: dict[str, object]
) -> None:
    class ScopedWatsonXConfig(IBMWatsonXAIConfig):
        pass

    ScopedWatsonXConfig(**supplied)

    assert ScopedWatsonXConfig.get_config() == stored


def test_init_stores_the_supplied_object_itself_rather_than_a_copy() -> None:
    class ScopedWatsonXConfig(IBMWatsonXAIConfig):
        pass

    stop_sequences = ["}", ")"]

    ScopedWatsonXConfig(stop_sequences=stop_sequences)

    assert ScopedWatsonXConfig.get_config()["stop_sequences"] is stop_sequences


def test_init_called_again_with_none_keeps_the_value_stored_earlier() -> None:
    class ScopedWatsonXConfig(IBMWatsonXAIConfig):
        pass

    ScopedWatsonXConfig(top_k=40)
    ScopedWatsonXConfig(top_k=None, top_p=0.5)

    assert ScopedWatsonXConfig.get_config() == {"top_k": 40, "top_p": 0.5, "kwargs": {}}
