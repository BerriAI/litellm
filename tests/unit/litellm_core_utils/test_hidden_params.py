from typing import Final

import pytest

from litellm.litellm_core_utils.hidden_params import (
    HIDDEN_PARAMS_ATTR,
    get_hidden_params,
    get_or_create_hidden_params,
    set_hidden_param,
    set_hidden_params,
)
from litellm.llms.openai.completion.transformation import OpenAITextCompletionConfig
from litellm.types.decisions import DecisionsResponse
from litellm.types.llms.base import HiddenParams
from litellm.types.utils import ModelResponse, TextChoices, TextCompletionResponse, Usage


def test_get_and_set_hidden_params_on_plain_object() -> None:
    class PlainResponse:
        def __init__(self) -> None:
            self._hidden_params = {"existing": True}

    response: Final = PlainResponse()
    stored: Final = get_hidden_params(response)
    assert stored is response._hidden_params

    replacement: Final = {"replacement": True}
    set_hidden_params(response, replacement)

    assert response._hidden_params is replacement
    assert get_hidden_params(response) is replacement


def test_get_and_set_hidden_params_on_dict() -> None:
    response: Final = {"_hidden_params": {"existing": True}}
    stored: Final = get_hidden_params(response)

    assert stored is response["_hidden_params"]

    replacement: Final = {"replacement": True}
    set_hidden_params(response, replacement)

    assert response["_hidden_params"] is replacement
    assert get_hidden_params(response) is replacement


def test_get_or_create_hidden_params_sets_empty_dict_on_plain_object() -> None:
    class PlainResponse:
        pass

    response: Final = PlainResponse()
    hidden_params: Final = get_or_create_hidden_params(response)

    assert response._hidden_params is hidden_params
    assert hidden_params == {}


def test_set_hidden_param_writes_existing_hidden_params_in_place() -> None:
    class PlainResponse:
        pass

    storage: Final = HiddenParams(response_cost=0.25)
    response: Final = PlainResponse()
    setattr(response, HIDDEN_PARAMS_ATTR, storage)

    set_hidden_param(response, "k", "value")

    assert getattr(response, HIDDEN_PARAMS_ATTR) is storage
    assert storage["k"] == "value"


def test_set_hidden_param_creates_dict_storage_when_missing_or_none() -> None:
    class PlainResponse:
        pass

    missing_response: Final = PlainResponse()
    none_response: Final = PlainResponse()
    setattr(none_response, HIDDEN_PARAMS_ATTR, None)

    set_hidden_param(missing_response, "k", "missing")
    set_hidden_param(none_response, "k", "none")

    assert getattr(missing_response, HIDDEN_PARAMS_ATTR) == {"k": "missing"}
    assert getattr(none_response, HIDDEN_PARAMS_ATTR) == {"k": "none"}


def test_set_hidden_param_writes_existing_dict_storage_in_place() -> None:
    class PlainResponse:
        pass

    storage: Final[dict[str, object]] = {"existing": True}
    response: Final = PlainResponse()
    setattr(response, HIDDEN_PARAMS_ATTR, storage)

    set_hidden_param(response, "k", "value")

    assert getattr(response, HIDDEN_PARAMS_ATTR) is storage
    assert storage == {"existing": True, "k": "value"}


def test_set_hidden_param_rejects_unsupported_storage_without_replacing_it() -> None:
    class PlainResponse:
        pass

    storage: Final = object()
    response: Final = PlainResponse()
    setattr(response, HIDDEN_PARAMS_ATTR, storage)

    with pytest.raises(TypeError, match="unsupported hidden params storage: object"):
        set_hidden_param(response, "k", "value")

    assert getattr(response, HIDDEN_PARAMS_ATTR) is storage


def test_get_or_create_hidden_params_rejects_hidden_params_storage_without_replacing_it() -> None:
    class PlainResponse:
        pass

    storage: Final = HiddenParams(response_cost=0.25)
    response: Final = PlainResponse()
    setattr(response, HIDDEN_PARAMS_ATTR, storage)

    with pytest.raises(TypeError, match="unsupported hidden params storage: HiddenParams"):
        get_or_create_hidden_params(response)

    assert getattr(response, HIDDEN_PARAMS_ATTR) is storage
    assert storage["response_cost"] == 0.25


def test_openai_text_completion_conversion_preserves_hidden_params_storage() -> None:
    source: Final = TextCompletionResponse(
        id="cmpl-test",
        object="text_completion",
        created=1,
        model="gpt-3.5-turbo-instruct",
        choices=[TextChoices(text="hello", index=0, logprobs=None, finish_reason="stop")],
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )
    response: Final = ModelResponse()
    storage: Final = HiddenParams(response_cost=0.25)
    setattr(response, HIDDEN_PARAMS_ATTR, storage)

    converted: Final = OpenAITextCompletionConfig().convert_to_chat_model_response_object(
        response_object=source,
        model_response_object=response,
    )

    assert converted is response
    assert getattr(response, HIDDEN_PARAMS_ATTR) is storage
    assert storage["original_response"] is source


def test_get_hidden_params_preserves_model_response_identity() -> None:
    response: Final = ModelResponse()

    assert get_hidden_params(response) is response.hidden_params


def test_get_hidden_params_returns_none_for_non_dict_storage() -> None:
    class PlainResponse:
        def __init__(self) -> None:
            self._hidden_params = HiddenParams(response_cost=0.25)

    response: Final = PlainResponse()

    assert get_hidden_params(response) is None


def test_set_hidden_params_replaces_frozen_decisions_response_private_attr() -> None:
    response: Final = DecisionsResponse(model="decider", answers={}, usage=None)
    replacement: Final = {"replacement": True}

    set_hidden_params(response, replacement)

    assert response.hidden_params is replacement
    assert response._hidden_params is replacement
