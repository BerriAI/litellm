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


def test_arize_space_and_api_key_headers():
    dest = build_destination("arize", {"arize_space_id": "S", "arize_api_key": "K"})
    assert dest is not None
    assert dest.endpoint == "https://otlp.arize.com/v1"
    assert dest.headers == {"space_id": "S", "api_key": "K"}


def test_weave_requires_endpoint_and_key():
    assert build_destination("weave_otel", {"wandb_api_key": "w"}) is None
    dest = build_destination(
        "weave_otel",
        {
            "wandb_api_key": "w",
            "weave_endpoint": "https://trace.wandb.ai/otel/v1/traces",
            "weave_project_id": "entity/project",
        },
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
