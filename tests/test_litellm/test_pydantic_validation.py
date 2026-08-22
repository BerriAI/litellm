import json
from typing import Final

import pytest
from pydantic import BaseModel, field_validator

import litellm
from litellm.llms.base_llm.base_utils import (
    _pydantic_model_json_schema,
    type_to_response_format_param,
)
from litellm.types.utils import LlmProviders, ModelResponse
from litellm.utils import (
    ProviderConfigManager,
    Rules,
    _apply_response_format_validation,
    _is_basemodel_class,
    _is_pydantic_basemodel_type,
    _should_preserve_pydantic_response_format,
    normalize_completion_response_format,
    post_call_processing,
    pre_process_non_default_params,
    process_response_format,
)


class MovieReview(BaseModel):
    title: str
    rating: int


class Actor(BaseModel):
    name: str


class Film(BaseModel):
    title: str
    lead: Actor


class AlphabeticReview(BaseModel):
    title: str
    rating: int

    @field_validator("title")
    @classmethod
    def title_must_be_alpha(cls, value: str) -> str:
        if not value.isalpha():
            raise TypeError("title must be alphabetic")
        return value


class TypeErrorReview(BaseModel):
    title: str
    rating: int

    @classmethod
    def model_validate_json(cls, json_data: str | bytes | bytearray, **kwargs):
        raise TypeError("custom validator failed")


class SchemaOnlyFormat:
    @classmethod
    def schema(cls) -> dict[str, object]:
        return {
            "title": "SchemaOnlyFormat",
            "type": "object",
            "properties": {"x": {"title": "X", "type": "string"}},
        }


class NoSchemaFormat:
    pass


def _mock_completion():
    pass


_mock_completion.__name__ = "completion"


def _make_response(content: str) -> ModelResponse:
    response = ModelResponse()
    response.choices[0].message.content = content
    return response


STRICT_SCHEMA: Final = {
    "type": "json_schema",
    "json_schema": {
        "name": "MovieReview",
        "schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "rating": {"type": "integer"},
            },
            "required": ["title", "rating"],
        },
    },
}


def test_process_response_format_converts_pydantic_v2_basemodel():
    processed: Final = process_response_format(MovieReview)

    assert processed is not None
    assert processed["type"] == "json_schema"
    json_schema: Final = processed["json_schema"]
    assert json_schema["name"] == "MovieReview"
    assert json_schema["strict"] is True
    schema: Final = json_schema["schema"]
    assert schema["type"] == "object"
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


def test_process_response_format_exits_early_for_none_bool_and_raw_dict():
    raw: Final = {"type": "json_object"}
    assert process_response_format(None) is None
    assert process_response_format(True) is None
    assert process_response_format(False) is None
    assert process_response_format("json") is None
    assert process_response_format(raw) == raw


def test_pydantic_v2_model_json_schema_helper():
    schema: Final = _pydantic_model_json_schema(MovieReview)
    assert schema["properties"]["title"]["type"] == "string"


def test_type_to_response_format_param_with_ref_template():
    processed: Final = type_to_response_format_param(Film, ref_template="/$defs/{model}")
    assert processed is not None
    assert processed["json_schema"]["name"] == "Film"


def test_pydantic_v1_schema_method_is_used_when_model_json_schema_absent():
    schema: Final = _pydantic_model_json_schema(SchemaOnlyFormat)
    assert schema["properties"]["x"]["type"] == "string"
    assert schema["title"] == "SchemaOnlyFormat"


def test_pydantic_schema_helper_raises_when_no_schema_methods():
    with pytest.raises(TypeError, match="Unsupported response_format type"):
        _pydantic_model_json_schema(NoSchemaFormat)


def test_pydantic_model_json_schema_accepts_ref_template():
    schema: Final = _pydantic_model_json_schema(Film, ref_template="/$defs/{model}")
    assert "title" in schema["properties"]
    assert "lead" in schema["properties"]


def test_is_pydantic_basemodel_type():
    assert _is_pydantic_basemodel_type(MovieReview) is True
    assert _is_pydantic_basemodel_type({"type": "json_object"}) is False
    assert _is_pydantic_basemodel_type(dict) is False
    assert _is_basemodel_class(MovieReview) is True
    assert _is_basemodel_class("json") is False


def test_is_pydantic_basemodel_type_swallows_issubclass_typeerror(monkeypatch):
    def _boom(cls, classinfo):
        raise TypeError("not a class")

    monkeypatch.setattr("builtins.issubclass", _boom)
    assert _is_pydantic_basemodel_type(MovieReview) is False
    assert _is_basemodel_class(MovieReview) is False


def test_strict_json_schema_failure_falls_back_to_model_json_schema(monkeypatch):
    def _boom(_model):
        raise TypeError("strict schema failed")

    monkeypatch.setattr(
        "litellm.llms.base_llm.base_utils._pydantic.to_strict_json_schema",
        _boom,
    )
    processed: Final = type_to_response_format_param(MovieReview)
    assert processed is not None
    assert processed["json_schema"]["schema"]["properties"]["title"]["type"] == "string"


def test_post_call_processing_raises_apierror_on_invalid_pydantic_json():
    with pytest.raises(litellm.APIError, match="Structured output") as exc:
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
    assert exc.value.status_code == 422


def test_post_call_processing_raises_apierror_on_pydantic_validation_error():
    with pytest.raises(litellm.APIError, match="Structured output") as exc:
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
    assert exc.value.status_code == 422


def test_custom_pydantic_validator_typeerror_becomes_apierror():
    with pytest.raises(litellm.APIError, match="Structured output") as exc:
        post_call_processing(
            _make_response(json.dumps({"title": "Inception", "rating": 9})),
            "gpt-4o",
            {
                "response_format": TypeErrorReview,
                "enable_json_schema_validation": True,
            },
            _mock_completion,
            Rules(),
        )
    assert exc.value.status_code == 422


def test_field_validator_typeerror_becomes_apierror():
    with pytest.raises(litellm.APIError, match="Structured output") as exc:
        post_call_processing(
            _make_response(json.dumps({"title": "Inception 2", "rating": 9})),
            "gpt-4o",
            {
                "response_format": AlphabeticReview,
                "enable_json_schema_validation": True,
            },
            _mock_completion,
            Rules(),
        )
    assert exc.value.status_code == 422


def test_invalid_json_jsonschema_validation_becomes_apierror():
    with pytest.raises(litellm.JSONSchemaValidationError) as exc:
        post_call_processing(
            _make_response("not-json"),
            "gpt-4o",
            {
                "response_format": STRICT_SCHEMA,
                "enable_json_schema_validation": True,
            },
            _mock_completion,
            Rules(),
        )
    assert exc.value.raw_response == "not-json"


def test_jsonschema_mismatch_becomes_apierror():
    payload: Final = json.dumps({"name": "test", "age": 25})
    with pytest.raises(litellm.JSONSchemaValidationError) as exc:
        post_call_processing(
            _make_response(payload),
            "gpt-4o",
            {
                "response_format": STRICT_SCHEMA,
                "enable_json_schema_validation": True,
            },
            _mock_completion,
            Rules(),
        )
    assert exc.value.raw_response == payload


def test_post_call_processing_accepts_valid_pydantic_response():  # test-quality-ok: unit test validation assertion
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


def test_post_call_skips_validation_for_non_schema_response_format():  # test-quality-ok: unit test validation assertion
    post_call_processing(
        _make_response("plain text"),
        "gpt-4o",
        {
            "response_format": "json",
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


def test_normalize_preserves_pydantic_class_for_gemini_and_vertex():
    gemini_preserved: Final = normalize_completion_response_format(
        MovieReview,
        model="gemini-2.5-flash",
        custom_llm_provider="gemini",
    )
    vertex_preserved: Final = normalize_completion_response_format(
        MovieReview,
        model="vertex_ai/gemini-2.5-pro",
        custom_llm_provider="vertex_ai",
    )
    prefix_preserved: Final = normalize_completion_response_format(
        MovieReview,
        model="gemini-2.5-pro",
        custom_llm_provider=None,
    )
    assert gemini_preserved is MovieReview
    assert vertex_preserved is MovieReview
    assert prefix_preserved is MovieReview


def test_normalize_converts_pydantic_class_for_openai():
    processed: Final = normalize_completion_response_format(
        MovieReview,
        model="gpt-4o",
        custom_llm_provider="openai",
    )
    assert isinstance(processed, dict)
    assert processed["type"] == "json_schema"
    assert processed["json_schema"]["name"] == "MovieReview"
    assert normalize_completion_response_format(None, model="gpt-4o") is None


def test_gdc_preserves_pydantic_via_vertex_params_flag():
    assert _should_preserve_pydantic_response_format("gdc", "ignored") is True
    assert _should_preserve_pydantic_response_format("vertex_ai_beta", "m") is True
    assert _should_preserve_pydantic_response_format(None, "gemini/gemini-2.5-flash") is True
    assert _should_preserve_pydantic_response_format(None, "vertex_ai_beta/gemini") is True
    assert _should_preserve_pydantic_response_format(None, "vertex_ai/gemini-2.5-pro") is True


def test_openai_and_bedrock_do_not_preserve_pydantic_class():
    assert _should_preserve_pydantic_response_format("openai", "gpt-4o") is False
    assert _should_preserve_pydantic_response_format("bedrock", "claude-4-sonnet") is False


def test_gemini_pre_process_keeps_compact_pydantic_schema():
    provider_config = ProviderConfigManager.get_provider_chat_config(
        model="gemini-2.5-flash",
        provider=LlmProviders.GEMINI,
    )
    processed: Final = pre_process_non_default_params(
        model="gemini-2.5-flash",
        passed_params={"model": "gemini-2.5-flash", "response_format": Film},
        special_params={},
        custom_llm_provider="gemini",
        additional_drop_params=None,
        provider_config=provider_config,
    )
    schema: Final = processed["response_format"]["json_schema"]["schema"]
    serialized: Final = json.dumps(schema)
    assert "$ref" in serialized or "$defs" in schema
    assert schema.get("additionalProperties") is not False


def test_vertex_pre_process_keeps_compact_pydantic_schema():
    provider_config = ProviderConfigManager.get_provider_chat_config(
        model="gemini-2.5-pro",
        provider=LlmProviders.VERTEX_AI,
    )
    processed: Final = pre_process_non_default_params(
        model="gemini-2.5-pro",
        passed_params={"model": "gemini-2.5-pro", "response_format": Film},
        special_params={},
        custom_llm_provider="vertex_ai",
        additional_drop_params=None,
        provider_config=provider_config,
    )
    schema: Final = processed["response_format"]["json_schema"]["schema"]
    serialized: Final = json.dumps(schema)
    assert "$ref" in serialized or "$defs" in schema


def test_apply_response_format_validation_none_is_noop():  # test-quality-ok: unit test validation assertion
    _apply_response_format_validation(
        response_format=None,
        model_response="not-json",
        model="gpt-4o",
    )
    _apply_response_format_validation(
        response_format=True,
        model_response="not-json",
        model="gpt-4o",
    )


def test_apply_response_format_validation_matching_pydantic_schema():  # test-quality-ok: unit test validation assertion
    payload: Final = json.dumps({"title": "Inception", "rating": 9})
    _apply_response_format_validation(
        response_format=MovieReview,
        model_response=payload,
        model="gpt-4o",
    )


def test_apply_response_format_validation_raw_json_schema_dict():  # test-quality-ok: unit test validation assertion
    payload: Final = json.dumps({"title": "Inception", "rating": 9})
    _apply_response_format_validation(
        response_format=STRICT_SCHEMA,
        model_response=payload,
        model="gpt-4o",
    )
    _apply_response_format_validation(
        response_format={"type": "json_object"},
        model_response="plain text",
        model="gpt-4o",
    )
    _apply_response_format_validation(
        response_format={"json_schema": "not-a-dict"},
        model_response="plain text",
        model="gpt-4o",
    )


def test_apply_response_format_validation_non_json_text_raises_apierror():
    with pytest.raises(litellm.JSONSchemaValidationError) as exc:
        _apply_response_format_validation(
            response_format=STRICT_SCHEMA,
            model_response="the movie was great",
            model="gpt-4o",
        )
    assert exc.value.raw_response == "the movie was great"
