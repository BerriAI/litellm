from typing import Final

import pytest

from litellm.litellm_core_utils.hidden_params import (
    HIDDEN_PARAMS_ATTR,
    get_hidden_params,
    get_hidden_params_storage,
    get_or_create_hidden_params,
    set_hidden_param,
    set_hidden_params,
)
from litellm.llms.openai.completion.transformation import OpenAITextCompletionConfig
from litellm.types.decisions import (
    DecisionsInputTokensDetails,
    DecisionsOutputTokensDetails,
    DecisionsResponse,
    DecisionsUsage,
)
from litellm.types.llms.base import HiddenParams
from litellm.types.utils import ModelResponse, TextChoices, TextCompletionResponse, Usage

_ZERO_USAGE: Final = DecisionsUsage(
    input_tokens=0,
    input_tokens_details=DecisionsInputTokensDetails(cached_tokens=0, cache_write_tokens=0),
    output_tokens=0,
    output_tokens_details=DecisionsOutputTokensDetails(reasoning_tokens=0),
    total_tokens=0,
)


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


def test_get_hidden_params_storage_preserves_supported_storage_identity() -> None:
    class PlainResponse:
        pass

    dict_storage: Final[dict[str, object]] = {"model_id": "dict-model"}
    model_storage: Final = HiddenParams(model_id="model-storage")
    dict_response: Final = PlainResponse()
    model_response: Final = PlainResponse()
    setattr(dict_response, HIDDEN_PARAMS_ATTR, dict_storage)
    setattr(model_response, HIDDEN_PARAMS_ATTR, model_storage)

    assert get_hidden_params_storage(dict_response) is dict_storage
    assert get_hidden_params_storage(model_response) is model_storage


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


def test_get_or_create_hidden_params_wraps_hidden_params_storage() -> None:
    class PlainResponse:
        pass

    storage: Final = HiddenParams(
        response_cost=0.25,
        headers={"x-ratelimit-remaining-requests": "5"},
    )
    response: Final = PlainResponse()
    setattr(response, HIDDEN_PARAMS_ATTR, storage)

    hidden_params: Final = get_or_create_hidden_params(response)

    assert getattr(response, HIDDEN_PARAMS_ATTR) is storage
    assert hidden_params["response_cost"] == 0.25
    assert hidden_params["headers"] == {"x-ratelimit-remaining-requests": "5"}
    assert "headers" in hidden_params
    assert "headers" in tuple(hidden_params)

    nested: Final = hidden_params.setdefault("nested", {})
    assert hidden_params.setdefault("nested", {}) is nested
    assert storage.nested is nested

    del hidden_params["headers"]
    assert "headers" not in hidden_params
    assert "headers" not in tuple(hidden_params)


def test_hidden_params_model_view_excludes_unset_fields() -> None:
    class PlainResponse:
        pass

    storage: Final = HiddenParams()
    response: Final = PlainResponse()
    setattr(response, HIDDEN_PARAMS_ATTR, storage)
    view: Final = get_or_create_hidden_params(response)

    assert not view
    assert list(view) == []
    assert "response_cost" not in view
    assert "_response_ms" not in view
    assert view.get("model_id", "d") == "d"
    assert "_response_ms" not in storage.model_fields_set

    with pytest.raises(KeyError, match="response_cost"):
        del view["response_cost"]

    view["model_id"] = "m"
    view["extra"] = 1

    assert "model_id" in view
    assert "extra" in view
    assert set(view) == {"model_id", "extra"}
    assert "model_id" in storage.model_fields_set
    assert "extra" in (storage.model_extra or {})
    assert storage.model_id == "m"
    assert storage.model_extra == {"extra": 1}

    fallback_response: Final = PlainResponse()
    setattr(fallback_response, HIDDEN_PARAMS_ATTR, HiddenParams())
    fallback_view: Final = get_or_create_hidden_params(fallback_response)
    merged_after_fallback: Final = {**fallback_view, "model_id": "m1"}
    merged_before_fallback: Final = {"model_id": "m1", **fallback_view}

    assert merged_after_fallback == {"model_id": "m1"}
    assert merged_before_fallback == {"model_id": "m1"}


def test_hidden_params_model_dump_includes_response_ms_when_excluding_unset() -> None:
    storage: Final = HiddenParams(response_cost=1.0)

    assert "_response_ms" in storage.model_dump(exclude_unset=True)


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
            self._hidden_params = object()

    response: Final = PlainResponse()

    assert get_hidden_params(response) is None
    assert get_hidden_params_storage(response) is None


def test_set_hidden_params_replaces_frozen_decisions_response_private_attr() -> None:
    response: Final = DecisionsResponse(model="decider", answers=(), usage=_ZERO_USAGE)
    replacement: Final = {"replacement": True}

    set_hidden_params(response, replacement)

    assert response.hidden_params is replacement
    assert response._hidden_params is replacement
