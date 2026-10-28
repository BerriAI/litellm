import base64
import json
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Final

import pytest
from pydantic import JsonValue

import litellm
from litellm.llms.vertex_ai.gemini.transformation import transform_request_body
from litellm.llms.vertex_ai.image_generation.vertex_gemini_transformation import VertexAIGeminiImageGenerationConfig
from litellm.llms.vertex_ai.image_generation.vertex_imagen_transformation import VertexAIImagenImageGenerationConfig
from litellm.llms.vertex_ai.rerank.transformation import VertexAIRerankConfig
from litellm.llms.vertex_ai.vertex_ai_partner_models.anthropic.transformation import VertexAIAnthropicConfig
from litellm.llms.vertex_ai.vertex_embeddings.transformation import VertexAITextEmbeddingConfig

FIELDS: Final = ["user_api_key_alias", "user_api_key_team_alias", "spend_logs_metadata"]
IDENTITY: Final = {"user_api_key_alias": "chargeback-key", "user_api_key_team_alias": "ai-platform"}
LABELS_HEADER: Final = "x-vertex-ai-labels"

LabelSource = Callable[[dict[str, JsonValue], dict[str, JsonValue]], object]


def gemini_labels(litellm_params: dict[str, JsonValue], optional_params: dict[str, JsonValue]) -> object:
    body: Final = transform_request_body(
        messages=[{"role": "user", "content": "hi"}],
        model="gemini-3.8-flash",
        optional_params=dict(optional_params),
        custom_llm_provider="vertex_ai",
        litellm_params=dict(litellm_params),
        cached_content=None,
    )
    return body.get("labels")


def embedding_labels(litellm_params: dict[str, JsonValue], optional_params: dict[str, JsonValue]) -> object:
    request: Final = VertexAITextEmbeddingConfig().transform_openai_request_to_vertex_embedding_request(
        input=["hi"], optional_params=dict(optional_params), model="text-embedding-005", litellm_params=litellm_params
    )
    return request.get("labels")


def imagen_labels(litellm_params: dict[str, JsonValue], optional_params: dict[str, JsonValue]) -> object:
    request: Final = VertexAIImagenImageGenerationConfig().transform_image_generation_request(
        model="imagen-4.0-generate-001",
        prompt="a cat",
        optional_params=dict(optional_params),
        litellm_params=dict(litellm_params),
        headers={},
    )
    return request.get("labels")


def gemini_image_labels(litellm_params: dict[str, JsonValue], optional_params: dict[str, JsonValue]) -> object:
    request: Final = VertexAIGeminiImageGenerationConfig().transform_image_generation_request(
        model="gemini-3.1-flash-image",
        prompt="a cat",
        optional_params=dict(optional_params),
        litellm_params=dict(litellm_params),
        headers={},
    )
    return request.get("labels")


def rerank_labels(litellm_params: dict[str, JsonValue], _: dict[str, JsonValue]) -> object:
    request: Final = VertexAIRerankConfig().transform_rerank_request(
        model="semantic-ranker-default@latest",
        optional_rerank_params={"query": "q", "documents": ["a", "b"]},
        headers={},
        litellm_params=dict(litellm_params),
    )
    return request.get("userLabels")


def encoded(labels: Mapping[str, str]) -> str:
    return base64.b64encode(json.dumps(dict(labels)).encode()).decode()


def claude_headers(
    litellm_params: dict[str, JsonValue],
    optional_params: dict[str, JsonValue],
    headers: Mapping[str, str] = MappingProxyType({}),
) -> Mapping[str, str]:
    outbound: Final = dict(headers)
    VertexAIAnthropicConfig().transform_request(
        model="claude-haiku-5-5",
        messages=[{"role": "user", "content": "hi"}],
        optional_params=dict(optional_params),
        litellm_params=dict(litellm_params),
        headers=outbound,
    )
    return outbound


def labels_header_names(headers: Mapping[str, str]) -> list[str]:
    return [name for name in headers if name.lower() == LABELS_HEADER]


def decoded_labels(headers: Mapping[str, str]) -> object:
    names: Final = labels_header_names(headers)
    return json.loads(base64.b64decode(headers[names[0]])) if names else None


def claude_labels(litellm_params: dict[str, JsonValue], optional_params: dict[str, JsonValue]) -> object:
    return decoded_labels(claude_headers(litellm_params, optional_params))


LABELLED_SURFACES: Final = [gemini_labels, embedding_labels, imagen_labels, gemini_image_labels]
REQUESTER_METADATA_SURFACES: Final = [*LABELLED_SURFACES, rerank_labels]
ALL_SURFACES: Final = [*REQUESTER_METADATA_SURFACES, claude_labels]
CALLER_LABEL_SURFACES: Final = [*LABELLED_SURFACES, claude_labels]


def metadata(key: str = "metadata", **fields: JsonValue) -> dict[str, JsonValue]:
    return {key: dict(fields)}


@pytest.mark.parametrize("setting", [None, []])
@pytest.mark.parametrize("surface", REQUESTER_METADATA_SURFACES)
def test_setting_off_sends_only_the_callers_labels(
    setting: object, surface: LabelSource, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(litellm, "vertex_request_label_fields", setting)
    params: Final = metadata(
        spend_logs_metadata={"cost-center": "cc-1"},
        requester_metadata={"user_api_key_team_alias": "caller-chosen"},
        **IDENTITY,
    )

    assert surface(params, {}) == {"user_api_key_team_alias": "caller-chosen"}
    assert surface(metadata(**IDENTITY), {}) is None


@pytest.mark.parametrize("setting", [None, []])
def test_setting_off_leaves_claude_labels_as_the_caller_sent_them(setting: object, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "vertex_request_label_fields", setting)
    params: Final = metadata(spend_logs_metadata={"cost-center": "cc-1"}, **IDENTITY)
    caller_header: Final = {"X-Vertex-AI-Labels": encoded({"user_api_key_team_alias": "Caller Chosen"})}

    assert claude_labels(params, {"labels": {"user_api_key_team_alias": "Caller Chosen"}}) == {
        "user_api_key_team_alias": "Caller Chosen"
    }
    assert claude_labels(params, {}) is None
    assert claude_headers(params, {"labels": {"app": "bot"}}, caller_header) == caller_header


@pytest.mark.parametrize("metadata_key", ["metadata", "litellm_metadata"])
@pytest.mark.parametrize("surface", ALL_SURFACES)
def test_identity_and_spend_logs_metadata_label_every_vertex_surface(
    metadata_key: str, surface: LabelSource, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(litellm, "vertex_request_label_fields", FIELDS)
    params: Final = metadata(metadata_key, spend_logs_metadata={"cost-center": "cc-1"}, **IDENTITY)

    assert surface(params, {}) == {**IDENTITY, "cost-center": "cc-1"}


@pytest.mark.parametrize("field_order", [FIELDS, list(reversed(FIELDS))])
@pytest.mark.parametrize(
    ("surface", "cap"),
    [
        (gemini_labels, 64),
        (embedding_labels, 64),
        (imagen_labels, 64),
        (gemini_image_labels, 64),
        (claude_labels, 32),
    ],
)
def test_identity_survives_a_caller_filling_every_label_slot(
    field_order: list[str], surface: LabelSource, cap: int, monkeypatch: pytest.MonkeyPatch
):
    """64 labels per call for Google models, 32 for partner models,
    https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/add-labels-to-api-calls (2026-10-28)"""
    monkeypatch.setattr(litellm, "vertex_request_label_fields", field_order)
    caller_labels: Final = {f"caller-{index:02d}": "v" for index in range(cap)}

    labels: Final = surface(metadata(**IDENTITY), {"labels": caller_labels})

    assert isinstance(labels, dict)
    assert len(labels) == cap
    assert {key: labels[key] for key in IDENTITY} == IDENTITY
    assert all(labels[key] == "v" for key in tuple(caller_labels)[: cap - len(IDENTITY)])


@pytest.mark.parametrize("surface", CALLER_LABEL_SURFACES)
def test_caller_cannot_write_or_override_the_identity_prefix_in_any_case(
    surface: LabelSource, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(litellm, "vertex_request_label_fields", FIELDS)
    forged: Final = {
        "user_api_key_team_alias": "forged",
        "User_API_Key_Team_Alias": "forged",
        " user_api_key_user_id": "forged",
        "app": "bot",
    }

    assert surface(metadata(**IDENTITY), {"labels": forged}) == {**IDENTITY, "app": "bot"}
    assert surface(metadata(user_api_key_alias="chargeback-key"), {"labels": forged}) == {
        "user_api_key_alias": "chargeback-key",
        "app": "bot",
    }
    assert surface(metadata(spend_logs_metadata={"USER_API_KEY_TEAM_ALIAS": "forged"}), {}) is None


def test_a_caller_set_claude_labels_header_cannot_forge_identity_and_is_replaced(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "vertex_request_label_fields", FIELDS)
    caller_header: Final = {"x-Vertex-AI-labels": encoded({"user_api_key_team_alias": "forged", "app": "bot"})}

    sent: Final = claude_headers(metadata(**IDENTITY), {"labels": {"ignored": "x"}}, caller_header)
    undecodable: Final = claude_headers(metadata(**IDENTITY), {}, {LABELS_HEADER: "%%"})

    assert len(labels_header_names(sent)) == 1
    assert decoded_labels(sent) == {**IDENTITY, "app": "bot"}
    assert len(labels_header_names(undecodable)) == 1
    assert decoded_labels(undecodable) == IDENTITY


@pytest.mark.parametrize("surface", ALL_SURFACES)
def test_identity_and_spend_logs_text_is_rewritten_to_labels_google_accepts(
    surface: LabelSource, monkeypatch: pytest.MonkeyPatch
):
    """Key and value rules from https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/add-labels-to-api-calls (2026-10-28)"""
    monkeypatch.setattr(litellm, "vertex_request_label_fields", FIELDS)
    spend_logs: Final = {
        "Owner": "Platform Team",
        "1st": "x",
        "_private": "x",
        "": "x",
        "dotted": "a.b",
        "too-long": "V" * 64,
        "K" * 64: "",
        "équipe": "Données-2",
        "部門": "営業",
        "numerals": "ⅻ½",
    }
    params: Final = metadata(
        user_api_key_alias="Chargeback Key",
        user_api_key_team_alias="AI Platform " + "x" * 60,
        spend_logs_metadata=spend_logs,
    )

    assert surface(params, {}) == {
        "user_api_key_alias": "chargeback_key",
        "user_api_key_team_alias": ("ai_platform_" + "x" * 60)[:63],
        "owner": "platform_team",
        "dotted": "a_b",
        "too-long": "v" * 63,
        "k" * 63: "",
        "équipe": "données-2",
        "部門": "営業",
        "numerals": "ⅻ½",
    }


@pytest.mark.parametrize("surface", CALLER_LABEL_SURFACES)
def test_caller_labels_keep_todays_pass_through_and_win_over_spend_logs_metadata(
    surface: LabelSource, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(litellm, "vertex_request_label_fields", FIELDS)
    params: Final = metadata(spend_logs_metadata={"cost-center": "cc-admin", "env": "prod"}, **IDENTITY)

    assert surface(params, {"labels": {"cost-center": "cc-caller", "Mixed": "Case", 1: "dropped"}}) == {
        **IDENTITY,
        "cost-center": "cc-caller",
        "Mixed": "Case",
        "env": "prod",
    }


@pytest.mark.parametrize("surface", ALL_SURFACES)
def test_only_allow_listed_identity_fields_and_opted_in_spend_logs_metadata_become_labels(
    surface: LabelSource, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(litellm, "vertex_request_label_fields", ["user_api_key_team_alias", "trace_id"])
    params: Final = metadata(trace_id="t-1", spend_logs_metadata={"cost-center": "cc-1"}, **IDENTITY)

    assert surface(params, {}) == {"user_api_key_team_alias": "ai-platform"}


def test_metadata_wins_over_litellm_metadata_for_the_same_identity_field(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "vertex_request_label_fields", FIELDS)
    params: Final[Mapping[str, JsonValue]] = {
        "metadata": {"user_api_key_team_alias": "from-metadata"},
        "litellm_metadata": {"user_api_key_team_alias": "from-litellm-metadata", "user_api_key_alias": "key-a"},
    }

    assert gemini_labels(dict(params), {}) == {
        "user_api_key_team_alias": "from-metadata",
        "user_api_key_alias": "key-a",
    }
