import re

import httpx
import pytest
import respx
from pydantic import ValidationError

import litellm
from litellm.llms.xai.common_utils import XAIModelInfo

MODELS_URL = "https://api.x.ai/v1/models"
MANY_MODELS = [{"id": f"grok-{index}"} for index in range(500)]


@pytest.fixture(autouse=True)
def _default_xai_api_base(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("XAI_API_BASE", raising=False)


def _get_models(payload: object) -> list[str]:
    with respx.mock:
        respx.get(MODELS_URL).mock(return_value=httpx.Response(200, json=payload))
        return XAIModelInfo().get_models(api_key="xai-test-key")


def test_get_models_prefixes_every_listed_model_id_in_provider_order():
    names = _get_models(
        {
            "object": "list",
            "data": [
                {"id": "grok-4", "object": "model", "created": 1752019200, "owned_by": "xai"},
                {"id": "grok-3-mini", "object": "model", "created": 1743724800, "owned_by": "xai"},
                {"id": ""},
            ],
        }
    )

    assert names == ["xai/grok-4", "xai/grok-3-mini", "xai/"]
    assert [type(name) for name in names] == [str, str, str]


@pytest.mark.parametrize("data", [[], "", {}])
def test_get_models_returns_no_names_for_an_empty_data_container(data: object):
    assert _get_models({"data": data}) == []


def test_get_models_sends_the_api_key_as_bearer_token():
    with respx.mock:
        route = respx.get(MODELS_URL).mock(return_value=httpx.Response(200, json={"data": [{"id": "grok-4"}]}))
        XAIModelInfo().get_models(api_key="xai-test-key")

    assert route.calls[0].request.headers["authorization"] == "Bearer xai-test-key"


@pytest.mark.parametrize(
    ("payload", "missing_key"),
    [
        ({"object": "list"}, "data"),
        ({"data": [{"object": "model"}]}, "id"),
        ({"data": [*MANY_MODELS[:403], {"object": "model"}, *MANY_MODELS[404:]]}, "id"),
    ],
)
def test_get_models_raises_key_error_naming_the_field_the_provider_left_out(payload: object, missing_key: str):
    with pytest.raises(KeyError) as exc_info:
        _get_models(payload)

    assert exc_info.value.args == (missing_key,)


@pytest.mark.parametrize(
    ("payload", "error_type"),
    [
        ([{"id": "grok-4"}], "dict_type"),
        ("data", "dict_type"),
        ({"data": None}, "iterable_type"),
        ({"data": 5}, "iterable_type"),
        ({"data": "grok-4"}, "dict_type"),
        ({"data": {"grok-4": {"id": "grok-4"}}}, "dict_type"),
        ({"data": ["grok-4"]}, "dict_type"),
        ({"data": [None]}, "dict_type"),
        ({"data": [{"id": None}]}, "string_type"),
        ({"data": [{"id": 12}]}, "string_type"),
        ({"data": [{"id": ["grok-4"]}]}, "string_type"),
        ({"data": [*MANY_MODELS[:403], "grok-403", *MANY_MODELS[404:]]}, "dict_type"),
        ({"data": [*MANY_MODELS[:429], {"id": 429}, *MANY_MODELS[430:]]}, "string_type"),
    ],
)
def test_get_models_rejects_a_malformed_listing_without_echoing_position_or_input(payload: object, error_type: str):
    with pytest.raises(ValidationError) as exc_info:
        _get_models(payload)

    assert [(error["type"], error["loc"]) for error in exc_info.value.errors()] == [(error_type, ())]
    assert "403" not in str(exc_info.value)
    assert "429" not in str(exc_info.value)
    assert "grok" not in str(exc_info.value)


def test_get_models_reports_the_upstream_status_and_body_on_an_http_error():
    with respx.mock:
        respx.get(MODELS_URL).mock(return_value=httpx.Response(401, text="bad key"))
        with pytest.raises(
            Exception, match=re.escape("Failed to fetch models from XAI. Status code: 401, Response: bad key")
        ):
            XAIModelInfo().get_models(api_key="xai-test-key")


@pytest.mark.parametrize(
    ("api_key", "payload", "expected"),
    [
        ("xai-listing-key", {"data": [{"id": "grok-4"}, {"id": "grok-3"}]}, ["xai/grok-4", "xai/grok-3"]),
        ("xai-bad-id-key", {"data": [{"id": "grok-4"}, {"id": 7}]}, []),
        ("xai-null-data-key", {"data": None}, []),
        ("xai-missing-data-key", {"object": "list"}, []),
        ("xai-list-body-key", [{"id": "grok-4"}], []),
    ],
)
def test_get_valid_models_lists_xai_models_and_falls_back_to_empty_on_a_malformed_listing(
    api_key: str, payload: object, expected: list[str]
):
    with respx.mock:
        respx.get(MODELS_URL).mock(return_value=httpx.Response(200, json=payload))
        models = litellm.get_valid_models(check_provider_endpoint=True, custom_llm_provider="xai", api_key=api_key)

    assert models == expected
