"""
Unit tests for litellm/llms/oci/common_utils.py.

Covers schema utilities, signing helpers, and credential resolution paths
that require no real OCI credentials or network calls.
"""

import pytest
from unittest.mock import MagicMock, patch

from litellm.llms.oci.common_utils import (
    OCI_API_VERSION,
    OCIError,
    OCIRequestWrapper,
    build_signature_string,
    enrich_cohere_param_description,
    get_oci_base_url,
    resolve_oci_credentials,
    resolve_oci_schema_anyof,
    resolve_oci_schema_refs,
    sanitize_oci_schema,
    sha256_base64,
    sign_oci_request,
    sign_with_oci_signer,
    validate_oci_environment,
)


# ---------------------------------------------------------------------------
# OCI_API_VERSION
# ---------------------------------------------------------------------------


def test_oci_api_version_constant():
    assert OCI_API_VERSION == "20231130"


# ---------------------------------------------------------------------------
# sha256_base64
# ---------------------------------------------------------------------------


def test_sha256_base64_known_value():
    import base64, hashlib

    data = b"hello"
    expected = base64.b64encode(hashlib.sha256(data).digest()).decode()
    assert sha256_base64(data) == expected


def test_sha256_base64_empty():
    result = sha256_base64(b"")
    assert isinstance(result, str)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# build_signature_string
# ---------------------------------------------------------------------------


def test_build_signature_string_request_target():
    headers = {"host": "example.com", "date": "Mon, 01 Jan 2024 00:00:00 GMT"}
    result = build_signature_string(
        "POST", "/20231130/actions/chat", headers, ["(request-target)", "host", "date"]
    )
    lines = result.split("\n")
    assert lines[0] == "(request-target): post /20231130/actions/chat"
    assert lines[1] == "host: example.com"
    assert lines[2] == "date: Mon, 01 Jan 2024 00:00:00 GMT"


def test_build_signature_string_method_lowercased():
    headers = {"host": "h"}
    result = build_signature_string("GET", "/path", headers, ["(request-target)"])
    assert result == "(request-target): get /path"


# ---------------------------------------------------------------------------
# OCIRequestWrapper.path_url
# ---------------------------------------------------------------------------


def test_request_wrapper_path_url_no_query():
    w = OCIRequestWrapper(
        method="POST",
        url="https://inference.generativeai.us-ashburn-1.oci.oraclecloud.com/20231130/actions/chat",
        headers={},
        body=b"",
    )
    assert w.path_url == "/20231130/actions/chat"


def test_request_wrapper_path_url_with_query():
    w = OCIRequestWrapper(
        method="GET",
        url="https://example.com/path?foo=bar&baz=1",
        headers={},
        body=b"",
    )
    assert w.path_url == "/path?foo=bar&baz=1"


# ---------------------------------------------------------------------------
# resolve_oci_credentials
# ---------------------------------------------------------------------------


def test_resolve_credentials_from_params():
    params = {
        "oci_region": "eu-frankfurt-1",
        "oci_user": "user1",
        "oci_fingerprint": "fp1",
        "oci_tenancy": "tenant1",
        "oci_key": "key_content",
        "oci_compartment_id": "comp1",
    }
    result = resolve_oci_credentials(params)
    assert result["oci_region"] == "eu-frankfurt-1"
    assert result["oci_user"] == "user1"
    assert result["oci_compartment_id"] == "comp1"


def test_resolve_credentials_env_fallback(monkeypatch):
    monkeypatch.setenv("OCI_REGION", "ap-tokyo-1")
    monkeypatch.setenv("OCI_USER", "env_user")
    monkeypatch.setenv("OCI_COMPARTMENT_ID", "env_comp")
    result = resolve_oci_credentials({})
    assert result["oci_region"] == "ap-tokyo-1"
    assert result["oci_user"] == "env_user"
    assert result["oci_compartment_id"] == "env_comp"


def test_resolve_credentials_region_default(monkeypatch):
    monkeypatch.delenv("OCI_REGION", raising=False)
    result = resolve_oci_credentials({})
    assert result["oci_region"] == "us-ashburn-1"


def test_resolve_credentials_params_override_env(monkeypatch):
    monkeypatch.setenv("OCI_REGION", "ap-tokyo-1")
    result = resolve_oci_credentials({"oci_region": "us-phoenix-1"})
    assert result["oci_region"] == "us-phoenix-1"


# ---------------------------------------------------------------------------
# get_oci_base_url
# ---------------------------------------------------------------------------


def test_get_oci_base_url_explicit_api_base():
    url = get_oci_base_url({}, api_base="https://custom.endpoint.com/")
    assert url == "https://custom.endpoint.com"


def test_get_oci_base_url_from_region():
    url = get_oci_base_url({"oci_region": "eu-frankfurt-1"})
    assert url == "https://inference.generativeai.eu-frankfurt-1.oci.oraclecloud.com"


@pytest.mark.parametrize(
    "bad_region",
    [
        "evil.com/#",  # SSRF: fragment truncates intended suffix in the URL
        "evil.com",  # dot escapes the {region} segment
        "us/ashburn",  # slash injects a path
        "US-ASHBURN-1",  # uppercase rejected
        "us ashburn 1",  # whitespace rejected
        "-leading-hyphen",  # must start with a letter
        "trailing-hyphen-",  # must not end with a hyphen
        "a" * 33,  # length cap
    ],
)
def test_get_oci_base_url_rejects_unsafe_region(bad_region):
    with pytest.raises(OCIError, match="Invalid oci_region"):
        get_oci_base_url({"oci_region": bad_region})


def test_get_oci_base_url_empty_region_falls_back_to_default():
    # An empty string trips the `or "us-ashburn-1"` fallback in
    # resolve_oci_credentials, so it's treated as "not supplied" — not as a
    # validation failure.
    url = get_oci_base_url({"oci_region": ""})
    assert url == "https://inference.generativeai.us-ashburn-1.oci.oraclecloud.com"


def test_get_oci_base_url_rejects_non_string_region():
    with pytest.raises(OCIError, match="Invalid oci_region"):
        get_oci_base_url({"oci_region": 12345})


@pytest.mark.parametrize(
    "good_region",
    ["us-ashburn-1", "us-chicago-1", "eu-frankfurt-1", "ap-tokyo-1", "sa-saopaulo-1"],
)
def test_get_oci_base_url_accepts_real_oci_regions(good_region):
    url = get_oci_base_url({"oci_region": good_region})
    assert url == f"https://inference.generativeai.{good_region}.oci.oraclecloud.com"


# ---------------------------------------------------------------------------
# validate_oci_environment
# ---------------------------------------------------------------------------


def test_validate_oci_environment_sets_defaults():
    headers = {}
    result = validate_oci_environment(headers, {})
    assert result["content-type"] == "application/json"
    assert "user-agent" in result


def test_validate_oci_environment_does_not_overwrite_existing():
    headers = {"content-type": "text/plain", "user-agent": "my-agent"}
    result = validate_oci_environment(headers, {})
    assert result["content-type"] == "text/plain"
    assert result["user-agent"] == "my-agent"


# ---------------------------------------------------------------------------
# sign_with_oci_signer — error paths
# ---------------------------------------------------------------------------


def test_sign_with_oci_signer_none_raises():
    with pytest.raises(ValueError, match="oci_signer cannot be None"):
        sign_with_oci_signer({}, {"oci_signer": None}, {}, "https://example.com")


def test_sign_with_oci_signer_exception_wrapped():
    bad_signer = MagicMock()
    bad_signer.do_request_sign.side_effect = RuntimeError("signing failed")
    with pytest.raises(OCIError, match="Failed to sign request"):
        sign_with_oci_signer(
            {}, {"oci_signer": bad_signer}, {"key": "val"}, "https://example.com"
        )


def test_sign_with_oci_signer_success():
    signer = MagicMock()
    signer.do_request_sign.return_value = None
    headers, body = sign_with_oci_signer(
        {}, {"oci_signer": signer}, {"key": "val"}, "https://example.com"
    )
    assert isinstance(body, bytes)
    signer.do_request_sign.assert_called_once()


# ---------------------------------------------------------------------------
# sign_oci_request — routing
# ---------------------------------------------------------------------------


def test_sign_oci_request_routes_to_signer():
    signer = MagicMock()
    signer.do_request_sign.return_value = None
    headers, body = sign_oci_request(
        {}, {"oci_signer": signer}, {}, "https://example.com"
    )
    signer.do_request_sign.assert_called_once()


def test_sign_oci_request_routes_to_manual_missing_creds():
    with pytest.raises(OCIError, match="Missing required OCI credentials"):
        sign_oci_request({}, {}, {}, "https://example.com")


# ---------------------------------------------------------------------------
# load_private_key_from_file — error paths (no real key needed)
# ---------------------------------------------------------------------------


def test_load_private_key_from_file_not_found():
    from litellm.llms.oci.common_utils import load_private_key_from_file

    with pytest.raises(FileNotFoundError, match="Private key file not found"):
        load_private_key_from_file("/nonexistent/path/key.pem")


def test_load_private_key_from_file_empty(tmp_path):
    from litellm.llms.oci.common_utils import load_private_key_from_file

    empty = tmp_path / "empty.pem"
    empty.write_text("")
    with pytest.raises(ValueError, match="Private key file is empty"):
        load_private_key_from_file(str(empty))


def test_load_private_key_from_file_os_error():
    from litellm.llms.oci.common_utils import load_private_key_from_file

    with patch("builtins.open", side_effect=OSError("permission denied")):
        with pytest.raises(OSError, match="Failed to read private key file"):
            load_private_key_from_file("/some/path/key.pem")


# ---------------------------------------------------------------------------
# resolve_oci_schema_refs
# ---------------------------------------------------------------------------


def test_resolve_schema_refs_basic():
    schema = {
        "$defs": {"Foo": {"type": "string"}},
        "properties": {"x": {"$ref": "#/$defs/Foo"}},
    }
    result = resolve_oci_schema_refs(schema)
    assert result["properties"]["x"] == {"type": "string"}
    assert "$defs" not in result


def test_resolve_schema_refs_external_ref_unchanged():
    schema = {"properties": {"x": {"$ref": "https://example.com/schema"}}}
    result = resolve_oci_schema_refs(schema)
    assert result["properties"]["x"] == {"$ref": "https://example.com/schema"}


def test_resolve_schema_refs_circular_breaks_cycle():
    schema = {
        "$defs": {"Node": {"properties": {"child": {"$ref": "#/$defs/Node"}}}},
        "properties": {"root": {"$ref": "#/$defs/Node"}},
    }
    result = resolve_oci_schema_refs(schema)
    # Should not raise; circular ref replaced with {"type": "object"}
    child = result["properties"]["root"]["properties"]["child"]
    assert child == {"type": "object"}


def test_resolve_schema_refs_no_defs():
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    result = resolve_oci_schema_refs(schema)
    assert result == schema


# ---------------------------------------------------------------------------
# resolve_oci_schema_anyof
# ---------------------------------------------------------------------------


def test_resolve_schema_anyof_optional_field():
    schema = {"anyOf": [{"type": "string"}, {"type": "null"}]}
    result = resolve_oci_schema_anyof(schema)
    assert result["type"] == "string"
    assert "anyOf" not in result


def test_resolve_schema_anyof_all_null_returns_empty():
    schema = {"anyOf": [{"type": "null"}, {"type": "null"}]}
    result = resolve_oci_schema_anyof(schema)
    # No non-null branch — anyOf stays or schema unchanged
    # The function only strips anyOf when there IS a non-null branch
    assert "anyOf" in result


def test_resolve_schema_anyof_no_anyof_unchanged():
    schema = {"type": "string", "description": "A name"}
    assert resolve_oci_schema_anyof(schema) == schema


def test_resolve_schema_anyof_nested():
    schema = {"properties": {"age": {"anyOf": [{"type": "integer"}, {"type": "null"}]}}}
    result = resolve_oci_schema_anyof(schema)
    assert result["properties"]["age"]["type"] == "integer"


# ---------------------------------------------------------------------------
# sanitize_oci_schema
# ---------------------------------------------------------------------------


def test_sanitize_schema_removes_title():
    schema = {"title": "MyModel", "type": "object", "properties": {}}
    result = sanitize_oci_schema(schema)
    assert "title" not in result


def test_sanitize_schema_removes_null_default():
    schema = {"type": "string", "default": None}
    result = sanitize_oci_schema(schema)
    assert "default" not in result


def test_sanitize_schema_keeps_non_null_default():
    schema = {"type": "string", "default": "hello"}
    result = sanitize_oci_schema(schema)
    assert result["default"] == "hello"


def test_sanitize_schema_type_any_becomes_object():
    schema = {"type": "any"}
    result = sanitize_oci_schema(schema)
    assert result["type"] == "object"


def test_sanitize_schema_type_list_picks_non_null():
    schema = {"type": ["string", "null"]}
    result = sanitize_oci_schema(schema)
    assert result["type"] == "string"


def test_sanitize_schema_type_list_all_null_becomes_string():
    schema = {"type": ["null"]}
    result = sanitize_oci_schema(schema)
    assert result["type"] == "string"


def test_sanitize_schema_array_gets_items():
    schema = {"type": "array"}
    result = sanitize_oci_schema(schema)
    assert result["items"] == {"type": "object"}


def test_sanitize_schema_array_keeps_existing_items():
    schema = {"type": "array", "items": {"type": "string"}}
    result = sanitize_oci_schema(schema)
    assert result["items"] == {"type": "string"}


def test_sanitize_schema_required_filters_missing_properties():
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "required": ["a", "b"],  # "b" not in properties
    }
    result = sanitize_oci_schema(schema)
    assert result["required"] == ["a"]


def test_sanitize_schema_required_non_list_becomes_empty():
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "required": "a",  # invalid: string instead of list
    }
    result = sanitize_oci_schema(schema)
    assert result["required"] == []


def test_sanitize_schema_list_input():
    schemas = [{"title": "A", "type": "string"}, {"title": "B", "type": "integer"}]
    result = sanitize_oci_schema(schemas)
    assert all("title" not in s for s in result)


# ---------------------------------------------------------------------------
# enrich_cohere_param_description
# ---------------------------------------------------------------------------


def test_enrich_description_enum():
    result = enrich_cohere_param_description("A color", {"enum": ["red", "blue"]})
    assert "Allowed values: ['red', 'blue']" in result


def test_enrich_description_format():
    result = enrich_cohere_param_description("A date", {"format": "date-time"})
    assert "Format: date-time" in result


def test_enrich_description_range_both():
    result = enrich_cohere_param_description("A number", {"minimum": 0, "maximum": 100})
    assert "Range: min=0, max=100" in result


def test_enrich_description_range_min_only():
    result = enrich_cohere_param_description("A number", {"minimum": 1})
    assert "Range: min=1" in result
    assert "max" not in result


def test_enrich_description_range_max_only():
    result = enrich_cohere_param_description("", {"maximum": 10})
    assert "Range: max=10" in result


def test_enrich_description_pattern():
    result = enrich_cohere_param_description("An ID", {"pattern": "^[a-z]+$"})
    assert "Pattern: ^[a-z]+$" in result


def test_enrich_description_all_constraints():
    result = enrich_cohere_param_description(
        "Val",
        {
            "enum": ["a"],
            "format": "uuid",
            "minimum": 0,
            "maximum": 1,
            "pattern": ".*",
        },
    )
    assert "Allowed values" in result
    assert "Format" in result
    assert "Range" in result
    assert "Pattern" in result


def test_enrich_description_no_constraints():
    result = enrich_cohere_param_description("Just a description", {})
    assert result == "Just a description"


def test_enrich_description_empty_description_no_constraints():
    result = enrich_cohere_param_description("", {})
    assert result == ""
