"""``build_destination`` maps an admin credential to a generic OTLP destination.

The point of these tests is that the resolution is backend-agnostic: Langfuse,
Arize, Weave, and any raw collector all resolve to an ``{endpoint, headers}``
the router exports through, and an incomplete credential resolves to nothing.
"""

import base64
import os
import sys

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.integrations.otel.presets.destinations import (
    OTEL_V2_DESTINATION_CALLBACKS,
    build_destination,
)


def test_langfuse_endpoint_derived_from_host_with_basic_auth():
    dest = build_destination(
        "langfuse_otel",
        {
            "langfuse_host": "https://cloud.langfuse.com",
            "langfuse_public_key": "pk-eu",
            "langfuse_secret_key": "sk-eu",
        },
    )
    assert dest is not None
    assert dest.endpoint == "https://cloud.langfuse.com/api/public/otel"
    scheme, b64 = dest.headers["Authorization"].split(" ", 1)
    assert scheme == "Basic"
    assert base64.b64decode(b64).decode() == "pk-eu:sk-eu"


def test_langfuse_bare_host_gets_https_and_path():
    dest = build_destination(
        "langfuse_otel",
        {
            "langfuse_host": "my-langfuse.internal",
            "langfuse_public_key": "pk",
            "langfuse_secret_key": "sk",
        },
    )
    assert dest is not None
    assert dest.endpoint == "https://my-langfuse.internal/api/public/otel"


def test_langfuse_without_host_defaults_to_us_cloud():
    dest = build_destination(
        "langfuse_otel",
        {"langfuse_public_key": "pk", "langfuse_secret_key": "sk"},
    )
    assert dest is not None
    assert dest.endpoint == "https://us.cloud.langfuse.com/api/public/otel"


def test_langfuse_incomplete_returns_none():
    assert build_destination("langfuse_otel", {"langfuse_public_key": "pk"}) is None


def test_arize_space_and_api_key_headers(monkeypatch):
    monkeypatch.delenv("ARIZE_PROJECT_NAME", raising=False)
    dest = build_destination("arize", {"arize_space_id": "S", "arize_api_key": "K"})
    assert dest is not None
    assert dest.endpoint == "https://otlp.arize.com/v1"
    assert dest.headers == {"space_id": "S", "api_key": "K"}


# --- per-backend Resource declaration -------------------------------------- #
#
# Each backend declares the Resource attributes its ingestion needs, in its own
# builder. Arize is the only first-class backend that routes by a Resource
# attribute (``model_id``); langfuse / weave / generic route by auth header and
# declare none. The shared ``destination_resource_attrs`` just reads whatever the
# builder put on the destination, so the model generalizes: a new backend that
# needs Resource-level routing only populates ``resource_attributes`` here.


def test_arize_project_from_credential_sets_resource_attrs(monkeypatch):
    monkeypatch.delenv("ARIZE_PROJECT_NAME", raising=False)
    dest = build_destination(
        "arize",
        {"arize_space_id": "S", "arize_api_key": "K", "arize_project_name": "team-x"},
    )
    assert dest is not None
    assert dest.resource_attributes == {
        "model_id": "team-x",
        "arize.project.name": "team-x",
    }


def test_arize_project_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("ARIZE_PROJECT_NAME", "env-proj")
    dest = build_destination("arize", {"arize_space_id": "S", "arize_api_key": "K"})
    assert dest is not None
    assert dest.resource_attributes == {
        "model_id": "env-proj",
        "arize.project.name": "env-proj",
    }


def test_arize_credential_project_wins_over_env(monkeypatch):
    monkeypatch.setenv("ARIZE_PROJECT_NAME", "env-proj")
    dest = build_destination(
        "arize",
        {
            "arize_space_id": "S",
            "arize_api_key": "K",
            "arize_project_name": "cred-proj",
        },
    )
    assert dest is not None
    assert dest.resource_attributes["model_id"] == "cred-proj"


def test_arize_no_project_anywhere_has_empty_resource_attrs(monkeypatch):
    monkeypatch.delenv("ARIZE_PROJECT_NAME", raising=False)
    dest = build_destination("arize", {"arize_space_id": "S", "arize_api_key": "K"})
    assert dest is not None
    assert dest.resource_attributes == {}


def test_header_routed_backends_declare_no_resource_attrs():
    """langfuse / weave / generic route the project via auth headers, so they
    declare no Resource attributes -- the generalization counterpart to arize."""
    langfuse = build_destination(
        "langfuse_otel",
        {"langfuse_public_key": "pk", "langfuse_secret_key": "sk"},
    )
    weave = build_destination(
        "weave_otel",
        {
            "wandb_api_key": "w",
            "weave_endpoint": "https://trace.wandb.ai/otel/v1/traces",
        },
    )
    generic = build_destination(
        "self_hosted", {"otel_endpoint": "https://collector:4318/v1/traces"}
    )
    for dest in (langfuse, weave, generic):
        assert dest is not None
        assert dest.resource_attributes == {}


def test_weave_requires_only_api_key_and_defaults_endpoint():
    # No API key -> nothing.
    assert build_destination("weave_otel", {}) is None
    # The API key alone is enough: Weave cloud's endpoint is fixed, so it defaults
    # to the cloud OTLP path and the endpoint field is optional.
    dest = build_destination(
        "weave_otel",
        {"wandb_api_key": "w", "weave_project_id": "entity/project"},
    )
    assert dest is not None
    assert dest.endpoint == "https://trace.wandb.ai/otel/v1/traces"
    assert dest.headers["project_id"] == "entity/project"
    assert "Authorization" in dest.headers


def test_generic_passthrough_covers_any_backend():
    dest = build_destination(
        "some_self_hosted_collector",
        {
            "otel_endpoint": "https://collector.internal:4318/v1/traces",
            "otel_headers": "x-api-key=abc,x-team=42",
        },
    )
    assert dest is not None
    assert dest.endpoint == "https://collector.internal:4318/v1/traces"
    assert dest.headers == {"x-api-key": "abc", "x-team": "42"}


def test_unknown_backend_without_generic_fields_returns_none():
    assert build_destination("mystery", {"foo": "bar"}) is None


def test_registry_lists_the_first_class_backends():
    assert OTEL_V2_DESTINATION_CALLBACKS == frozenset(
        {"langfuse_otel", "arize", "weave_otel"}
    )


def test_endpoint_whitespace_is_trimmed():
    # A stray leading/trailing space in the endpoint (an easy create-form slip)
    # makes a malformed OTLP URL the exporter rejects with a 404, so values are
    # trimmed before the destination is built.
    dest = build_destination(
        "some_collector",
        {"otel_endpoint": "  https://collector.internal:4318/v1/traces  "},
    )
    assert dest is not None
    assert dest.endpoint == "https://collector.internal:4318/v1/traces"


def test_weave_endpoint_completed_to_otel_path():
    # Weave's OTLP path is /otel/v1/traces, not the bare /v1/traces the generic
    # exporter would append; a host must be completed here or the export 404s.
    # Idempotent when the full path or the /otel prefix is already supplied.
    for given, expected in (
        ("https://trace.wandb.ai", "https://trace.wandb.ai/otel/v1/traces"),
        ("https://trace.wandb.ai/otel", "https://trace.wandb.ai/otel/v1/traces"),
        (
            "https://trace.wandb.ai/otel/v1/traces",
            "https://trace.wandb.ai/otel/v1/traces",
        ),
    ):
        dest = build_destination(
            "weave_otel", {"wandb_api_key": "w", "weave_endpoint": given}
        )
        assert dest is not None
        assert dest.endpoint == expected
