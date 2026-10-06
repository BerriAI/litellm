from typing import Final

from litellm.litellm_core_utils.hidden_params import (
    get_hidden_params,
    get_or_create_hidden_params,
    set_hidden_params,
)
from litellm.types.decisions import DecisionsResponse
from litellm.types.llms.base import HiddenParams
from litellm.types.utils import ModelResponse


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


def test_get_or_create_hidden_params_sets_empty_dict_on_plain_object() -> None:
    class PlainResponse:
        pass

    response: Final = PlainResponse()
    hidden_params: Final = get_or_create_hidden_params(response)

    assert response._hidden_params is hidden_params
    assert hidden_params == {}


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
