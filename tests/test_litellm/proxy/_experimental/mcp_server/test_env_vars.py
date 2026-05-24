"""Unit tests for the per-server / per-user env-var resolver.

Covers the pure-Python pieces of `env_vars.py` — interpolation, missing-var
detection, deep-link generation. DB-backed helpers (`store_user_env_vars`,
`get_user_env_vars`) are exercised in the integration suite.
"""

import os
from typing import Dict

import pytest

from litellm.proxy._experimental.mcp_server.env_vars import (
    EnvVarDefinition,
    MissingEnvVarsError,
    collect_placeholders,
    interpolate_headers,
    interpolate_value,
    missing_required,
    parse_env_var_definitions,
    resolve_values,
)


class TestParseEnvVarDefinitions:
    def test_none_returns_empty(self):
        assert parse_env_var_definitions(None) == []

    def test_empty_list_returns_empty(self):
        assert parse_env_var_definitions([]) == []

    def test_list_of_dicts(self):
        defs = parse_env_var_definitions(
            [
                {"name": "TOKEN", "scope": "per_user"},
                {"name": "HOST", "scope": "instance", "value": "db.corp"},
            ]
        )
        assert len(defs) == 2
        assert defs[0].name == "TOKEN"
        assert defs[0].scope == "per_user"
        assert defs[1].value == "db.corp"

    def test_json_string_parsed(self):
        defs = parse_env_var_definitions('[{"name": "TOKEN", "scope": "per_user"}]')
        assert len(defs) == 1
        assert defs[0].name == "TOKEN"

    def test_invalid_json_string_returns_empty(self):
        assert parse_env_var_definitions("not json {{{") == []

    def test_malformed_entries_dropped_silently(self):
        defs = parse_env_var_definitions(
            [
                {"name": "GOOD", "scope": "per_user"},
                "not a dict",
                {"missing_scope": "BAD"},
                None,
            ]
        )
        # Only the GOOD entry should survive.
        names = [d.name for d in defs]
        assert names == ["GOOD"]

    def test_pre_typed_entries_pass_through(self):
        original = EnvVarDefinition(name="TOKEN", scope="per_user")
        defs = parse_env_var_definitions([original])
        assert defs == [original]


class TestCollectPlaceholders:
    def test_no_placeholders(self):
        assert collect_placeholders(["plain string", "no vars here"]) == []

    def test_single_placeholder(self):
        assert collect_placeholders(["Bearer ${TOKEN}"]) == ["TOKEN"]

    def test_multiple_placeholders_dedup(self):
        result = collect_placeholders(
            [
                "Bearer ${TOKEN}",
                "${HOST}:${PORT}/${TOKEN}",  # TOKEN dup
            ]
        )
        # Order preserved, no dupes
        assert result == ["TOKEN", "HOST", "PORT"]

    def test_ignores_none_and_empty(self):
        assert collect_placeholders([None, "", "${A}"]) == ["A"]

    def test_lowercase_placeholders_ignored(self):
        # Regex only matches UPPER_SNAKE_CASE — matches the UI's validator.
        assert collect_placeholders(["${lowercase}", "${MixedCase}"]) == []


class TestMissingRequired:
    def test_no_referenced_returns_empty(self):
        defs = [EnvVarDefinition(name="A", scope="per_user")]
        assert missing_required(defs, {}, referenced=[]) == []

    def test_per_user_with_value_not_missing(self):
        defs = [EnvVarDefinition(name="TOKEN", scope="per_user")]
        assert missing_required(defs, {"TOKEN": "abc"}, referenced=["TOKEN"]) == []

    def test_per_user_empty_value_is_missing(self):
        defs = [EnvVarDefinition(name="TOKEN", scope="per_user")]
        assert missing_required(defs, {"TOKEN": ""}, referenced=["TOKEN"]) == ["TOKEN"]

    def test_instance_missing_is_not_reported(self):
        # Admin misconfiguration (an instance var with no value) is an
        # admin-side problem, not a user-side one. The resolver simply leaves
        # the placeholder in place; we don't surface it as a deep-link prompt.
        defs = [EnvVarDefinition(name="HOST", scope="instance", value=None)]
        assert missing_required(defs, {}, referenced=["HOST"]) == []

    def test_unknown_referenced_name_ignored(self):
        # If a placeholder references a var that isn't declared, it's not
        # "missing per-user" — it's just unresolvable. Don't shadow the
        # admin-misconfig path.
        defs = [EnvVarDefinition(name="OTHER", scope="per_user")]
        assert missing_required(defs, {}, referenced=["UNRELATED"]) == []


class TestResolveValues:
    def test_per_user_wins_over_instance(self):
        defs = [
            EnvVarDefinition(name="X", scope="instance", value="admin-default"),
            EnvVarDefinition(name="X", scope="per_user"),
        ]
        result = resolve_values(defs, {"X": "user-value"}, referenced=["X"])
        assert result == {"X": "user-value"}

    def test_instance_used_when_no_per_user(self):
        defs = [EnvVarDefinition(name="HOST", scope="instance", value="db.corp")]
        assert resolve_values(defs, {}, referenced=["HOST"]) == {"HOST": "db.corp"}

    def test_unreferenced_vars_skipped(self):
        # Critical: a stray missing per-user value must NOT block a request
        # that doesn't reference it.
        defs = [
            EnvVarDefinition(name="USED", scope="instance", value="ok"),
            EnvVarDefinition(name="UNUSED", scope="per_user"),
        ]
        assert resolve_values(defs, {}, referenced=["USED"]) == {"USED": "ok"}

    def test_instance_with_empty_value_skipped(self):
        defs = [EnvVarDefinition(name="X", scope="instance", value="")]
        assert resolve_values(defs, {}, referenced=["X"]) == {}


class TestInterpolateHeaders:
    def test_substitutes_known_placeholders(self):
        headers = {"Authorization": "Bearer ${TOKEN}", "X-Host": "${HOST}"}
        resolved = {"TOKEN": "abc123", "HOST": "db.corp"}
        assert interpolate_headers(headers, resolved) == {
            "Authorization": "Bearer abc123",
            "X-Host": "db.corp",
        }

    def test_leaves_unresolved_placeholders_in_place(self):
        # Important: don't silently strip — caller should have already
        # detected missing vars and short-circuited with a useful error.
        headers = {"X-Token": "${MISSING}"}
        assert interpolate_headers(headers, {}) == {"X-Token": "${MISSING}"}

    def test_none_passes_through(self):
        assert interpolate_headers(None, {"X": "y"}) is None

    def test_returns_new_dict_not_mutated(self):
        original = {"X": "${A}"}
        interpolate_headers(original, {"A": "1"})
        assert original == {"X": "${A}"}

    def test_multiple_placeholders_in_one_value(self):
        headers = {
            "X-Conn": "${PROTO}://${USER}:${PASS}@${HOST}",
        }
        resolved = {
            "PROTO": "postgresql",
            "USER": "alice",
            "PASS": "s3cr3t",
            "HOST": "db.corp:5432",
        }
        assert interpolate_headers(headers, resolved) == {
            "X-Conn": "postgresql://alice:s3cr3t@db.corp:5432",
        }


class TestInterpolateValue:
    def test_substitutes(self):
        assert interpolate_value("Bearer ${T}", {"T": "x"}) == "Bearer x"

    def test_none_passes_through(self):
        assert interpolate_value(None, {"T": "x"}) is None

    def test_no_placeholder(self):
        assert interpolate_value("Bearer hardcoded", {"T": "x"}) == "Bearer hardcoded"


class TestMissingEnvVarsError:
    def test_deep_link_uses_proxy_base_url(self, monkeypatch):
        monkeypatch.delenv("PROXY_UI_BASE_URL", raising=False)
        monkeypatch.setenv("PROXY_BASE_URL", "https://proxy.example.com")
        e = MissingEnvVarsError(
            server_alias="github_corp",
            server_name="GitHub (corp)",
            missing=["TOKEN"],
        )
        assert (
            e.deep_link()
            == "https://proxy.example.com/ui/tools/mcp-servers?fill_fields=github_corp"
        )

    def test_deep_link_trims_trailing_slash(self, monkeypatch):
        monkeypatch.delenv("PROXY_UI_BASE_URL", raising=False)
        monkeypatch.setenv("PROXY_BASE_URL", "https://proxy.example.com/")
        e = MissingEnvVarsError(
            server_alias="alias",
            server_name=None,
            missing=["X"],
        )
        assert (
            e.deep_link()
            == "https://proxy.example.com/ui/tools/mcp-servers?fill_fields=alias"
        )

    def test_deep_link_falls_back_to_localhost(self, monkeypatch):
        monkeypatch.delenv("PROXY_UI_BASE_URL", raising=False)
        monkeypatch.delenv("PROXY_BASE_URL", raising=False)
        e = MissingEnvVarsError(
            server_alias="x",
            server_name=None,
            missing=["X"],
        )
        assert (
            e.deep_link() == "http://localhost:4000/ui/tools/mcp-servers?fill_fields=x"
        )

    def test_deep_link_respects_proxy_ui_base_url_override(self, monkeypatch):
        # Dev workflow: UI on :3000, proxy on :4000. The UI override gets
        # used verbatim — no `/ui` prefix is appended.
        monkeypatch.setenv("PROXY_UI_BASE_URL", "http://localhost:3000")
        monkeypatch.setenv("PROXY_BASE_URL", "http://localhost:4000")
        e = MissingEnvVarsError(
            server_alias="dev_server",
            server_name=None,
            missing=["TOKEN"],
        )
        assert (
            e.deep_link()
            == "http://localhost:3000/tools/mcp-servers?fill_fields=dev_server"
        )

    def test_user_message_includes_server_name_and_link(self, monkeypatch):
        monkeypatch.delenv("PROXY_UI_BASE_URL", raising=False)
        monkeypatch.setenv("PROXY_BASE_URL", "https://proxy.test")
        e = MissingEnvVarsError(
            server_alias="x",
            server_name="My Server",
            missing=["CORP_PASSWORD", "API_TOKEN"],
        )
        msg = e.to_user_message()
        assert 'Cannot connect to MCP server "My Server"' in msg
        assert "CORP_PASSWORD" in msg
        assert "API_TOKEN" in msg
        assert "https://proxy.test/ui/tools/mcp-servers?fill_fields=x" in msg

    def test_user_message_falls_back_to_alias_when_no_name(self):
        e = MissingEnvVarsError(
            server_alias="my-alias",
            server_name=None,
            missing=["X"],
        )
        assert 'Cannot connect to MCP server "my-alias"' in e.to_user_message()


class TestEndToEndResolveFlow:
    """The pattern the manager follows: parse defs → collect refs → check
    missing → resolve → interpolate. Locked in here so refactors of the
    individual helpers can't drift the contract.
    """

    def _resolve_or_raise(
        self,
        static_headers: Dict[str, str],
        defs_list,
        per_user_values: Dict[str, str],
        server_alias: str = "demo",
    ) -> Dict[str, str]:
        defs = parse_env_var_definitions(defs_list)
        referenced = collect_placeholders(static_headers.values())
        missing = missing_required(defs, per_user_values, referenced=referenced)
        if missing:
            raise MissingEnvVarsError(
                server_alias=server_alias,
                server_name=server_alias,
                missing=missing,
            )
        resolved = resolve_values(defs, per_user_values, referenced=referenced)
        return interpolate_headers(static_headers, resolved)

    def test_happy_path_per_user_and_instance(self):
        headers = self._resolve_or_raise(
            static_headers={
                "Authorization": "Bearer ${CORP_TOKEN}",
                "X-Tenant": "${TENANT_ID}",
            },
            defs_list=[
                {"name": "CORP_TOKEN", "scope": "per_user"},
                {"name": "TENANT_ID", "scope": "instance", "value": "tenant-42"},
            ],
            per_user_values={"CORP_TOKEN": "user-secret"},
        )
        assert headers == {
            "Authorization": "Bearer user-secret",
            "X-Tenant": "tenant-42",
        }

    def test_missing_per_user_raises_with_deep_link(self):
        with pytest.raises(MissingEnvVarsError) as excinfo:
            self._resolve_or_raise(
                static_headers={"Authorization": "Bearer ${CORP_TOKEN}"},
                defs_list=[{"name": "CORP_TOKEN", "scope": "per_user"}],
                per_user_values={},
                server_alias="github_corp",
            )
        e = excinfo.value
        assert e.missing == ["CORP_TOKEN"]
        # Deep link respects the alias, not a slugified server_name.
        assert "fill_fields=github_corp" in e.deep_link()

    def test_no_placeholders_skips_lookup_entirely(self):
        # Server has no `${X}` placeholders → resolver should pass through
        # cleanly even when defs are absent / per-user values are empty.
        headers = self._resolve_or_raise(
            static_headers={"X-Plain": "literal"},
            defs_list=[],
            per_user_values={},
        )
        assert headers == {"X-Plain": "literal"}

    def test_unused_missing_per_user_does_not_block(self):
        # Server declares a per-user var that nothing references → no error,
        # the request goes through.
        headers = self._resolve_or_raise(
            static_headers={"X-Plain": "literal"},
            defs_list=[{"name": "UNUSED", "scope": "per_user"}],
            per_user_values={},
        )
        assert headers == {"X-Plain": "literal"}
