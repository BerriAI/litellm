"""Vertex Model Garden: OpenAPI base URL for publisher/model ids vs per-endpoint path."""

import pytest

from litellm.llms.vertex_ai.vertex_model_garden.main import (
    _vertex_model_garden_model_id_in_json_body,
    create_vertex_url,
)


@pytest.mark.parametrize(
    "model,expect_openapi_base",
    [
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
    assert _vertex_model_garden_model_id_in_json_body("xai/grok-4.1-fast-reasoning") is True
    assert _vertex_model_garden_model_id_in_json_body("5464397967697903616") is False
