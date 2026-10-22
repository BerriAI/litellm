"""
Unit tests for litellm/llms/oci/common_utils.py.

Covers schema utilities, signing helpers, and credential resolution paths
that require no real OCI credentials or network calls.
"""

import sys
import types
from types import MappingProxyType
from typing import Final
from unittest.mock import MagicMock, patch

import pytest

from litellm.llms.oci.common_utils import (
    _OCI_REALM_DOMAINS,
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
    import base64
    import hashlib

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
    result = build_signature_string("POST", "/20231130/actions/chat", headers, ["(request-target)", "host", "date"])
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


@pytest.mark.parametrize(
    "api_base",
    [
        "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/20231130/actions/chat",
        "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/20231130/actions/chat/",
        "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/20231130/actions/embedText",
    ],
)
def test_get_oci_base_url_strips_trailing_action_path(api_base):
    assert get_oci_base_url({}, api_base=api_base) == "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"


@pytest.mark.usefixtures("without_oci_sdk", "isolated_region_metadata")
def test_get_oci_base_url_from_region():
    url = get_oci_base_url({"oci_region": "eu-frankfurt-1"})
    assert url == "https://inference.generativeai.eu-frankfurt-1.oci.oraclecloud.com"


@pytest.mark.parametrize(
    "region",
    [
        "evil.com/#",
        "evil.com",
        "us-ashburn-1/../attacker",
        "ATTACKER",
        "-leading-hyphen",
        "trailing-hyphen-",
        "a",
        "a" * 33,
        "us ashburn 1",
        "us_ashburn_1",
    ],
)
def test_get_oci_base_url_rejects_unsafe_region(region):
    with pytest.raises(OCIError, match="Invalid OCI region"):
        get_oci_base_url({"oci_region": region})


@pytest.mark.usefixtures("without_oci_sdk", "isolated_region_metadata")
def test_get_oci_base_url_empty_region_falls_back_to_default(monkeypatch):
    monkeypatch.delenv("OCI_REGION", raising=False)
    url = get_oci_base_url({"oci_region": ""})
    assert url == "https://inference.generativeai.us-ashburn-1.oci.oraclecloud.com"


@pytest.mark.parametrize(
    "region",
    [
        "us-ashburn-1",
        "eu-frankfurt-1",
        "ap-tokyo-1",
        "us-chicago-1",
        "us-phoenix-1",
        "ap",
    ],
)
@pytest.mark.usefixtures("without_oci_sdk", "isolated_region_metadata")
def test_get_oci_base_url_accepts_valid_region(region):
    url = get_oci_base_url({"oci_region": region})
    assert url == f"https://inference.generativeai.{region}.oci.oraclecloud.com"


_NON_COMMERCIAL_REALMS: Final = (
    ("oc2", "us-luke-1", "oraclegovcloud.com"),
    ("oc3", "us-gov-ashburn-1", "oraclegovcloud.com"),
    ("oc4", "uk-gov-london-1", "oraclegovcloud.uk"),
    ("oc19", "eu-frankfurt-2", "oraclecloud.eu"),
)
_UNKNOWN_REGION: Final = "xx-nowhere-1"
_UNKNOWN_REALM_COMPARTMENT: Final = "ocid1.compartment.oc99..aaaaaaaaexample"
_UNKNOWN_REGION_METADATA: Final = '{"realmKey": "OCX", "realmDomainComponent": "example.test", "regionKey": "XNW", "regionIdentifier": "xx-nowhere-1"}'


def _compartment(realm):
    return f"ocid1.compartment.{realm}..aaaaaaaaexample"


def _params(region: str, compartment_id: object = None) -> MappingProxyType[str, object]:
    return MappingProxyType({"oci_region": region, "oci_compartment_id": compartment_id})


@pytest.fixture
def without_oci_sdk(monkeypatch):
    monkeypatch.setitem(sys.modules, "oci", None)
    monkeypatch.setitem(sys.modules, "oci.regions", None)


@pytest.fixture
def isolated_region_metadata(monkeypatch, tmp_path):
    monkeypatch.delenv("OCI_REGION_METADATA", raising=False)
    monkeypatch.delenv("OCI_COMPARTMENT_ID", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


def test_realm_table_matches_installed_sdk():
    # Realm domains per the OCI Python SDK's oci.regions_definitions.REALMS (v2.187.0, checked 2026-09-27)
    definitions: Final = pytest.importorskip("oci.regions_definitions")
    assert (
        MappingProxyType({realm: definitions.REALMS.get(realm) for realm in _OCI_REALM_DOMAINS}) == _OCI_REALM_DOMAINS
    )


@pytest.mark.usefixtures("isolated_region_metadata")
@pytest.mark.parametrize(("realm", "region", "second_level_domain"), _NON_COMMERCIAL_REALMS)
def test_get_oci_base_url_resolves_realm_from_region_via_sdk(realm, region, second_level_domain):
    pytest.importorskip("oci.regions")
    # Realm domains per the OCI Python SDK's oci.regions_definitions (v2.187.0, checked 2026-09-27)
    url: Final = get_oci_base_url(_params(region))
    assert url == f"https://inference.generativeai.{region}.oci.{second_level_domain}"


@pytest.mark.usefixtures("without_oci_sdk", "isolated_region_metadata")
@pytest.mark.parametrize(("realm", "region", "second_level_domain"), _NON_COMMERCIAL_REALMS)
def test_get_oci_base_url_resolves_realm_from_compartment_ocid(realm, region, second_level_domain):
    url: Final = get_oci_base_url(_params(region, _compartment(realm)))
    assert url == f"https://inference.generativeai.{region}.oci.{second_level_domain}"


@pytest.mark.usefixtures("without_oci_sdk", "isolated_region_metadata")
def test_get_oci_base_url_resolves_realm_from_compartment_env(monkeypatch):
    monkeypatch.setenv("OCI_COMPARTMENT_ID", _compartment("oc2"))
    url: Final = get_oci_base_url(_params("us-luke-1"))
    assert url == "https://inference.generativeai.us-luke-1.oci.oraclegovcloud.com"


@pytest.mark.usefixtures("without_oci_sdk", "isolated_region_metadata")
def test_get_oci_base_url_reads_realm_key_case_insensitively():
    url: Final = get_oci_base_url(_params("us-luke-1", _compartment("OC2")))
    assert url == "https://inference.generativeai.us-luke-1.oci.oraclegovcloud.com"


@pytest.mark.usefixtures("without_oci_sdk", "isolated_region_metadata")
def test_get_oci_base_url_keeps_commercial_compartment_commercial():
    url: Final = get_oci_base_url(_params("us-chicago-1", _compartment("oc1")))
    assert url == "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"


@pytest.mark.usefixtures("without_oci_sdk", "isolated_region_metadata")
@pytest.mark.parametrize("compartment_id", (None, "not-an-ocid", _UNKNOWN_REALM_COMPARTMENT, 42))
def test_get_oci_base_url_without_sdk_defaults_to_commercial_when_realm_unknown(compartment_id):
    url: Final = get_oci_base_url(_params(_UNKNOWN_REGION, compartment_id))
    assert url == f"https://inference.generativeai.{_UNKNOWN_REGION}.oci.oraclecloud.com"


@pytest.mark.usefixtures("without_oci_sdk", "isolated_region_metadata")
def test_get_oci_base_url_compartment_realm_wins_over_region_metadata(monkeypatch):
    monkeypatch.setenv(
        "OCI_REGION_METADATA", '{"regionIdentifier": "us-luke-1", "realmDomainComponent": "example.test"}'
    )
    url: Final = get_oci_base_url(_params("us-luke-1", _compartment("oc2")))
    assert url == "https://inference.generativeai.us-luke-1.oci.oraclegovcloud.com"


@pytest.mark.usefixtures("without_oci_sdk", "isolated_region_metadata")
def test_get_oci_base_url_without_sdk_uses_region_metadata_env(monkeypatch):
    monkeypatch.setenv("OCI_REGION_METADATA", _UNKNOWN_REGION_METADATA)
    url: Final = get_oci_base_url(_params(_UNKNOWN_REGION, _UNKNOWN_REALM_COMPARTMENT))
    assert url == f"https://inference.generativeai.{_UNKNOWN_REGION}.oci.example.test"


@pytest.mark.usefixtures("without_oci_sdk", "isolated_region_metadata")
def test_get_oci_base_url_without_sdk_region_metadata_leaves_other_regions_commercial(monkeypatch):
    monkeypatch.setenv("OCI_REGION_METADATA", _UNKNOWN_REGION_METADATA)
    url: Final = get_oci_base_url(_params("us-chicago-1"))
    assert url == "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"


@pytest.mark.usefixtures("without_oci_sdk")
def test_get_oci_base_url_without_sdk_uses_regions_config_file(isolated_region_metadata):
    oci_dir: Final = isolated_region_metadata / ".oci"
    oci_dir.mkdir()
    (oci_dir / "regions-config.json").write_text(f"[{_UNKNOWN_REGION_METADATA}]")
    url: Final = get_oci_base_url(_params(_UNKNOWN_REGION))
    assert url == f"https://inference.generativeai.{_UNKNOWN_REGION}.oci.example.test"


@pytest.mark.usefixtures("without_oci_sdk")
def test_get_oci_base_url_without_sdk_keeps_valid_regions_config_entries_next_to_a_bad_one(isolated_region_metadata):
    oci_dir: Final = isolated_region_metadata / ".oci"
    oci_dir.mkdir()
    (oci_dir / "regions-config.json").write_text(
        f'[{{"regionIdentifier": "us-langley-1"}}, {_UNKNOWN_REGION_METADATA}]'
    )
    url: Final = get_oci_base_url(_params(_UNKNOWN_REGION))
    assert url == f"https://inference.generativeai.{_UNKNOWN_REGION}.oci.example.test"


@pytest.mark.usefixtures("without_oci_sdk")
@pytest.mark.parametrize("content", (b"\xff\xfe\x00[", b'{"regionIdentifier": "xx-nowhere-1"}', b"not json"))
def test_get_oci_base_url_without_sdk_ignores_unusable_regions_config_file(isolated_region_metadata, content):
    oci_dir: Final = isolated_region_metadata / ".oci"
    oci_dir.mkdir()
    (oci_dir / "regions-config.json").write_bytes(content)
    url: Final = get_oci_base_url(_params(_UNKNOWN_REGION))
    assert url == f"https://inference.generativeai.{_UNKNOWN_REGION}.oci.oraclecloud.com"


@pytest.mark.usefixtures("without_oci_sdk", "isolated_region_metadata")
@pytest.mark.parametrize(
    "metadata",
    (
        '{"regionIdentifier": "xx-nowhere-1", "realmDomainComponent": "evil.com/#"}',
        '{"regionIdentifier": "xx-nowhere-1", "realmDomainComponent": "-internal"}',
        '{"regionIdentifier": "xx-nowhere-1"}',
        "not json",
    ),
)
def test_get_oci_base_url_without_sdk_ignores_invalid_region_metadata(monkeypatch, metadata):
    monkeypatch.setenv("OCI_REGION_METADATA", metadata)
    url: Final = get_oci_base_url(_params(_UNKNOWN_REGION))
    assert url == f"https://inference.generativeai.{_UNKNOWN_REGION}.oci.oraclecloud.com"


def _fake_oci_regions(endpoint_for=None):
    module: Final = types.ModuleType("oci.regions")
    if endpoint_for is not None:
        module.endpoint_for = endpoint_for
    return module


@pytest.mark.usefixtures("isolated_region_metadata")
def test_get_oci_base_url_uses_sdk_region_registry_when_realm_unknown(monkeypatch):
    endpoint_for: Final = MagicMock(
        side_effect=lambda service, region, service_endpoint_template: service_endpoint_template.format(
            region=region, secondLevelDomain="example.test"
        )
    )
    monkeypatch.setitem(sys.modules, "oci", types.ModuleType("oci"))
    monkeypatch.setitem(sys.modules, "oci.regions", _fake_oci_regions(endpoint_for))
    url: Final = get_oci_base_url(_params(_UNKNOWN_REGION, _UNKNOWN_REALM_COMPARTMENT))
    assert url == f"https://inference.generativeai.{_UNKNOWN_REGION}.oci.example.test"
    endpoint_for.assert_called_once_with(
        "generative_ai_inference",
        region=_UNKNOWN_REGION,
        service_endpoint_template="https://inference.generativeai.{region}.oci.{secondLevelDomain}",
    )


@pytest.mark.usefixtures("isolated_region_metadata")
def test_get_oci_base_url_skips_sdk_region_registry_when_compartment_realm_known(monkeypatch):
    def endpoint_for(service, region, service_endpoint_template):
        raise AssertionError("registry consulted")

    monkeypatch.setitem(sys.modules, "oci", types.ModuleType("oci"))
    monkeypatch.setitem(sys.modules, "oci.regions", _fake_oci_regions(endpoint_for))
    url: Final = get_oci_base_url(_params("us-luke-1", _compartment("oc2")))
    assert url == "https://inference.generativeai.us-luke-1.oci.oraclegovcloud.com"


@pytest.mark.usefixtures("isolated_region_metadata")
def test_get_oci_base_url_prefers_sdk_region_registry_over_hand_parsed_metadata(monkeypatch):
    def endpoint_for(service, region, service_endpoint_template):
        return service_endpoint_template.format(region=region, secondLevelDomain="sdk.test")

    monkeypatch.setitem(sys.modules, "oci", types.ModuleType("oci"))
    monkeypatch.setitem(sys.modules, "oci.regions", _fake_oci_regions(endpoint_for))
    monkeypatch.setenv("OCI_REGION_METADATA", _UNKNOWN_REGION_METADATA)
    url: Final = get_oci_base_url(_params(_UNKNOWN_REGION))
    assert url == f"https://inference.generativeai.{_UNKNOWN_REGION}.oci.sdk.test"


@pytest.mark.usefixtures("isolated_region_metadata")
def test_get_oci_base_url_falls_back_to_metadata_when_sdk_registry_lacks_endpoint_for(monkeypatch):
    monkeypatch.setitem(sys.modules, "oci", types.ModuleType("oci"))
    monkeypatch.setitem(sys.modules, "oci.regions", _fake_oci_regions())
    monkeypatch.setenv("OCI_REGION_METADATA", _UNKNOWN_REGION_METADATA)
    url: Final = get_oci_base_url(_params(_UNKNOWN_REGION))
    assert url == f"https://inference.generativeai.{_UNKNOWN_REGION}.oci.example.test"


@pytest.mark.usefixtures("without_oci_sdk", "isolated_region_metadata")
@pytest.mark.parametrize(
    ("metadata", "second_level_domain"),
    (
        ('{"regionIdentifier": "XX-NOWHERE-1", "realmDomainComponent": "Example.Test"}', "example.test"),
        ('{"regionIdentifier": "xx-nowhere-1", "realmDomainComponent": "internal"}', "internal"),
    ),
)
def test_get_oci_base_url_without_sdk_normalizes_region_metadata_like_the_sdk(
    monkeypatch, metadata, second_level_domain
):
    monkeypatch.setenv("OCI_REGION_METADATA", metadata)
    url: Final = get_oci_base_url(_params(_UNKNOWN_REGION))
    assert url == f"https://inference.generativeai.{_UNKNOWN_REGION}.oci.{second_level_domain}"


@pytest.mark.usefixtures("without_oci_sdk")
def test_get_oci_base_url_without_sdk_tolerates_unresolvable_home(monkeypatch):
    def no_passwd_entry(uid):
        raise KeyError(uid)

    monkeypatch.delenv("HOME", raising=False)
    monkeypatch.setattr("pwd.getpwuid", no_passwd_entry)
    url: Final = get_oci_base_url(_params(_UNKNOWN_REGION))
    assert url == f"https://inference.generativeai.{_UNKNOWN_REGION}.oci.oraclecloud.com"


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
        sign_with_oci_signer({}, {"oci_signer": bad_signer}, {"key": "val"}, "https://example.com")


def test_sign_with_oci_signer_success():
    signer = MagicMock()
    signer.do_request_sign.return_value = None
    headers, body = sign_with_oci_signer({}, {"oci_signer": signer}, {"key": "val"}, "https://example.com")
    assert isinstance(body, bytes)
    signer.do_request_sign.assert_called_once()


# ---------------------------------------------------------------------------
# sign_oci_request — routing
# ---------------------------------------------------------------------------


def test_sign_oci_request_routes_to_signer():
    signer = MagicMock()
    signer.do_request_sign.return_value = None
    headers, body = sign_oci_request({}, {"oci_signer": signer}, {}, "https://example.com")
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
