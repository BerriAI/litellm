import json
from typing import Final
from unittest.mock import patch

import pytest
from pydantic import BaseModel, ValidationError

import litellm
from litellm.llms.base_llm.base_utils import _pydantic_model_json_schema, type_to_response_format_param
from litellm.types.utils import ModelResponse
from litellm.utils import Rules, post_call_processing, process_response_format


class MovieReview(BaseModel):
    title: str
    rating: int


def _mock_completion():
    pass


_mock_completion.__name__ = "completion"


def _make_response(content: str) -> ModelResponse:
    response = ModelResponse()
    response.choices[0].message.content = content
    return response


def test_process_response_format_converts_pydantic_v2_basemodel():
    processed: Final = process_response_format(MovieReview)

    assert processed is not None
    assert processed["type"] == "json_schema"
    json_schema: Final = processed["json_schema"]
    assert json_schema["name"] == "MovieReview"
    assert json_schema["strict"] is True
    schema: Final = json_schema["schema"]
    assert schema["type"] == "object"
    assert "title" in schema["properties"]
    assert "rating" in schema["properties"]
    assert schema["properties"]["title"]["type"] == "string"
    assert schema["properties"]["rating"]["type"] == "integer"


def test_process_response_format_passthrough_none_and_dict():
    existing: Final = {
        "type": "json_schema",
        "json_schema": {
            "name": "MovieReview",
            "schema": {"type": "object", "properties": {"title": {"type": "string"}}},
        },
    }
    assert process_response_format(None) is None
    assert process_response_format(existing)["json_schema"]["name"] == "MovieReview"


def test_pydantic_v1_schema_fallback_when_model_json_schema_missing():
    class LegacyShape(BaseModel):
        x: str

    def _v1_schema() -> dict:
        return {
            "title": "LegacyShape",
            "type": "object",
            "properties": {"x": {"title": "X", "type": "string"}},
        }

    with patch.object(LegacyShape, "model_json_schema", None):
        with patch.object(LegacyShape, "schema", staticmethod(_v1_schema)):
            schema: Final = _pydantic_model_json_schema(LegacyShape)

    assert schema["properties"]["x"]["type"] == "string"
    assert schema["title"] == "LegacyShape"


def test_type_to_response_format_param_falls_back_when_strict_schema_fails():
    with patch(
        "litellm.llms.base_llm.base_utils._pydantic.to_strict_json_schema",
        side_effect=ValidationError.from_exception_data("MovieReview", []),
    ):
        processed: Final = type_to_response_format_param(MovieReview)

    assert processed is not None
    assert processed["json_schema"]["schema"]["properties"]["title"]["type"] == "string"


def test_post_call_processing_raises_apierror_on_invalid_pydantic_json():
    with pytest.raises(litellm.APIError, match="Structured output"):
        post_call_processing(
            _make_response("not-json"),
            "gpt-4o",
            {
                "response_format": MovieReview,
                "enable_json_schema_validation": True,
            },
            _mock_completion,
            Rules(),
        )


def test_post_call_processing_raises_apierror_on_pydantic_validation_error():
    with pytest.raises(litellm.APIError, match="Structured output"):
        post_call_processing(
            _make_response(json.dumps({"title": "Inception", "rating": "nine"})),
            "gpt-4o",
            {
                "response_format": MovieReview,
                "enable_json_schema_validation": True,
            },
            _mock_completion,
            Rules(),
        )


def test_post_call_processing_accepts_valid_pydantic_response():
    post_call_processing(
        _make_response(json.dumps({"title": "Inception", "rating": 9})),
        "gpt-4o",
        {
            "response_format": MovieReview,
            "enable_json_schema_validation": True,
        },
        _mock_completion,
        Rules(),
    )


def test_completion_converts_pydantic_response_format_with_mock_response():
    response: Final = litellm.completion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "review"}],
        response_format=MovieReview,
        mock_response=json.dumps({"title": "Inception", "rating": 9}),
    )
    assert response.choices[0].message.content is not None
    payload: Final = json.loads(response.choices[0].message.content)
    assert payload["title"] == "Inception"
    assert payload["rating"] == 9
