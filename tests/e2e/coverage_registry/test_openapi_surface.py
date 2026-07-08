from __future__ import annotations

import pytest

from coverage_registry.openapi_surface import (
    InvalidMarker,
    MarkerEntry,
    Surface,
    build_openapi_surface,
    compute_surface_coverage,
    marker_entry_from_pytest_marker,
    render_loki,
)


def test_openapi_surface_expands_llm_endpoint_params_by_provider() -> None:
    surface = build_openapi_surface(
        {
            "paths": {
                "/chat/completions": {
                    "post": {
                        "tags": ["chat/completions"],
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "model": {"type": "string"},
                                            "messages": {"type": "array"},
                                            "stream": {"type": "boolean"},
                                        },
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    )

    assert (
        Surface("core_llms", "/chat/completions", "POST", "openai", "stream")
        in surface.surfaces
    )
    assert (
        Surface("core_llms", "/chat/completions", "POST", "anthropic", "stream")
        in surface.surfaces
    )
    assert (
        Surface("core_llms", "/chat/completions", "POST", "proxy", "stream")
        not in surface.surfaces
    )


def test_surface_coverage_reports_uncovered_concrete_provider_params() -> None:
    surface = build_openapi_surface(
        {
            "paths": {
                "/chat/completions": {
                    "post": {
                        "tags": ["chat/completions"],
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "model": {"type": "string"},
                                            "messages": {"type": "array"},
                                        },
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    )
    report = compute_surface_coverage(
        surface,
        (
            MarkerEntry(
                nodeid="tests/e2e/test_chat.py::test_chat",
                endpoint="/chat/completions",
                method="POST",
                providers=("openai",),
                params=("model",),
            ),
        ),
    )

    assert report.covered == 1
    assert (
        Surface("core_llms", "/chat/completions", "POST", "openai", "messages")
        in report.uncovered
    )
    assert (
        Surface("core_llms", "/chat/completions", "POST", "anthropic", "model")
        in report.uncovered
    )


def test_invalid_marker_rejects_fake_provider_and_param_labels() -> None:
    mark = pytest.mark.e2e_coverage(
        endpoint="/chat/completions",
        provider="multiple",
        params=("tool-calling",),
    ).mark

    parsed, error = marker_entry_from_pytest_marker("node", mark)

    assert parsed is None
    assert error is not None
    assert "providers must be concrete" in error


def test_invalid_marker_for_unknown_openapi_param_is_reported() -> None:
    surface = build_openapi_surface(
        {
            "paths": {
                "/chat/completions": {
                    "post": {
                        "tags": ["chat/completions"],
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {"model": {"type": "string"}},
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    )

    report = compute_surface_coverage(
        surface,
        (
            MarkerEntry(
                nodeid="node",
                endpoint="/chat/completions",
                method="POST",
                providers=("openai",),
                params=("not_in_openapi",),
            ),
        ),
    )

    assert report.invalid_markers == (
        InvalidMarker(
            nodeid="node",
            reason=(
                "marker does not match OpenAPI surface: POST /chat/completions "
                "provider=openai param=not_in_openapi"
            ),
        ),
    )


def test_schema_gap_is_reported_for_provider_endpoint_without_body_params() -> None:
    surface = build_openapi_surface(
        {
            "paths": {
                "/v1/messages": {
                    "post": {
                        "tags": ["anthropic_passthrough"],
                        "operationId": "anthropic_response_v1_messages_post",
                        "responses": {"200": {"description": "ok"}},
                    }
                }
            }
        }
    )

    assert {gap.reason for gap in surface.schema_gaps} == {
        "missing JSON requestBody in OpenAPI",
        "no request params exposed in OpenAPI",
    }


def test_loki_renderer_emits_distinct_openapi_coverage_lines() -> None:
    surface = build_openapi_surface(
        {
            "paths": {
                "/chat/completions": {
                    "post": {
                        "tags": ["chat/completions"],
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {"model": {"type": "string"}},
                                    }
                                }
                            }
                        },
                    }
                }
            }
        }
    )
    report = compute_surface_coverage(
        surface,
        (
            MarkerEntry(
                nodeid="node",
                endpoint="/chat/completions",
                method="POST",
                providers=("openai",),
                params=("model",),
            ),
        ),
    )

    output = render_loki(report)

    assert "OPENAPI_COVERAGE_TOTAL percent=" in output
    assert "OPENAPI_COVERAGE_MODULE module=core_llms" in output
    assert "OPENAPI_COVERAGE_UNCOVERED module=core_llms" in output
    assert not any(line.startswith("COVERAGE_TOTAL ") for line in output.splitlines())
