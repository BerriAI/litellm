"""Vertex Model Garden: OpenAPI base URL for publisher/model ids vs per-endpoint path."""

import json
from pathlib import Path

import pytest

import litellm
from litellm.llms.vertex_ai.common_utils import (
    VertexAIModelRoute,
    get_vertex_ai_model_route,
)
from litellm.llms.vertex_ai.vertex_model_garden.main import (
    _vertex_model_garden_model_id_in_json_body,
    create_vertex_url,
)


@pytest.mark.parametrize(
    "model,expect_openapi_base",
    [
        ("google/gemma-4-26b-a4b-it-maas", True),
        ("xai/grok-4.1-fast-reasoning", True),
        ("openai/foo/bar", True),
        ("5464397967697903616", False),
        ("gpt-oss-20b-maas", False),
    ],
)
def test_create_vertex_url_openapi_vs_deployed_endpoint(
    model: str, expect_openapi_base: bool
) -> None:
    url = create_vertex_url(
        vertex_location="us-central1",
        vertex_project="my-project",
        stream=False,
        model=model,
    )
    if expect_openapi_base:
        assert "/v1/projects/my-project/locations/us-central1/endpoints/openapi" in url
    else:
        assert (
            "/v1beta1/projects/my-project/locations/us-central1/endpoints/"
            f"{model}" in url
        )
        assert "openapi" not in url


def test_model_id_in_json_body_heuristic() -> None:
    assert (
        _vertex_model_garden_model_id_in_json_body("google/gemma-4-26b-a4b-it-maas")
        is True
    )
    assert (
        _vertex_model_garden_model_id_in_json_body("xai/grok-4.1-fast-reasoning")
        is True
    )
    assert _vertex_model_garden_model_id_in_json_body("5464397967697903616") is False


def test_gemma_4_26b_pricing_metadata_uses_vertex_model_garden_route():
    model = "google/gemma-4-26b-a4b-it-maas"
    litellm_model = f"vertex_ai/{model}"

    assert get_vertex_ai_model_route(model=model) == VertexAIModelRoute.MODEL_GARDEN
    assert (
        get_vertex_ai_model_route(model="gemma/gemma-3-12b-it")
        == VertexAIModelRoute.GEMMA
    )

    repo_root = Path(litellm.__file__).resolve().parents[1]
    model_cost_files = [
        repo_root / "model_prices_and_context_window.json",
        repo_root / "litellm" / "model_prices_and_context_window_backup.json",
    ]

    for model_cost_file in model_cost_files:
        model_cost = json.loads(model_cost_file.read_text(encoding="utf-8"))

        assert "vertex_ai/gemma/gemma-4-26b-a4b-it-maas" not in model_cost
        model_info = model_cost[litellm_model]

        assert model_info["max_input_tokens"] == 262144
        assert model_info["max_output_tokens"] == 128000
        assert model_info["max_tokens"] == 128000
        assert model_info["supports_function_calling"] is True
        assert model_info["supports_tool_choice"] is True
