import sys
from types import ModuleType, SimpleNamespace

from litellm.proxy import _lazy_openapi_snapshot as snapshot_module
from litellm.proxy._lazy_openapi_snapshot import (
    EXIT_CANNOT_VERIFY,
    EXIT_DRIFTED,
    EXIT_IN_SYNC,
    REGEN_COMMAND,
    _normalize_operation_ids,
    _register_lazy_feature,
    check_snapshot,
    drifted_features,
    serialize_snapshot,
)


def test_generate_snapshot_uses_shared_operation_id_reservations(monkeypatch):
    from litellm.proxy import _lazy_openapi_snapshot

    route_a = SimpleNamespace(path="/feature-a/items")
    route_b = SimpleNamespace(path="/feature-b/items")
    fake_app = SimpleNamespace(
        title="LiteLLM test",
        version="0.0.0",
        routes=[route_a, route_b],
    )

    fake_feature_a_module = ModuleType("fake_feature_a")
    fake_feature_b_module = ModuleType("fake_feature_b")
    monkeypatch.setitem(sys.modules, "fake_feature_a", fake_feature_a_module)
    monkeypatch.setitem(sys.modules, "fake_feature_b", fake_feature_b_module)

    fake_lazy_features_module = ModuleType("litellm.proxy._lazy_features")
    fake_lazy_features_module.LAZY_FEATURES = [
        SimpleNamespace(
            name="feature-a",
            module_path="fake_feature_a",
            path_prefixes=("/feature-a",),
            register_fn=lambda app, module: None,
        ),
        SimpleNamespace(
            name="feature-b",
            module_path="fake_feature_b",
            path_prefixes=("/feature-b",),
            register_fn=lambda app, module: None,
        ),
    ]
    monkeypatch.setitem(sys.modules, "litellm.proxy._lazy_features", fake_lazy_features_module)

    def fake_get_openapi(title, version, routes):
        path = routes[0].path
        return {
            "paths": {path: {"get": {"operationId": "shared_operation_id_get"}}},
            "components": {"schemas": {"Example": {"type": "object"}}},
        }

    def fake_ensure_unique_openapi_operation_ids(schema, reserved_operation_ids):
        for path_item in schema["paths"].values():
            operation = path_item["get"]
            operation_id = operation["operationId"]
            if operation_id in reserved_operation_ids:
                operation_id = f"{operation_id}_2"
                operation["operationId"] = operation_id
            reserved_operation_ids.add(operation_id)
        return schema

    fake_proxy_server_module = ModuleType("litellm.proxy.proxy_server")
    fake_proxy_server_module.app = fake_app
    fake_proxy_server_module.ensure_unique_openapi_operation_ids = fake_ensure_unique_openapi_operation_ids
    monkeypatch.setitem(sys.modules, "litellm.proxy.proxy_server", fake_proxy_server_module)
    monkeypatch.setattr("fastapi.openapi.utils.get_openapi", fake_get_openapi)

    fragments = _lazy_openapi_snapshot.generate_snapshot()

    assert fragments["feature-a"]["paths"]["/feature-a/items"]["get"]["operationId"] == "shared_operation_id_get"
    assert fragments["feature-b"]["paths"]["/feature-b/items"]["get"]["operationId"] == "shared_operation_id_get_2"
    assert fragments["feature-a"]["paths"]["/feature-a/items"]["get"]["tags"] == ["feature-a"]
    assert fragments["feature-b"]["paths"]["/feature-b/items"]["get"]["tags"] == ["feature-b"]


def test_normalize_operation_ids_uses_each_http_method():
    paths = {
        "/proxy/{endpoint}": {
            "delete": {"operationId": "proxy_route_proxy__endpoint__put"},
            "get": {"operationId": "proxy_route_proxy__endpoint__put"},
            "post": {"operationId": "proxy_route_proxy__endpoint__put"},
            "put": {"operationId": "proxy_route_proxy__endpoint__put"},
        }
    }

    _normalize_operation_ids(paths)

    operations = paths["/proxy/{endpoint}"]
    assert operations["delete"]["operationId"] == "proxy_route_proxy__endpoint__delete"
    assert operations["get"]["operationId"] == "proxy_route_proxy__endpoint__get"
    assert operations["post"]["operationId"] == "proxy_route_proxy__endpoint__post"
    assert operations["put"]["operationId"] == "proxy_route_proxy__endpoint__put"


def test_normalize_operation_ids_preserves_custom_ids():
    paths = {
        "/proxy/{endpoint}": {
            "get": {"operationId": "custom_operation"},
            "post": {"operationId": "custom_operation"},
        }
    }

    _normalize_operation_ids(paths)

    operations = paths["/proxy/{endpoint}"]
    assert operations["get"]["operationId"] == "custom_operation"
    assert operations["post"]["operationId"] == "custom_operation"


class TestDriftedFeatures:
    def test_reports_changed_added_and_removed_fragments(self):
        committed = {"kept": {"paths": {}}, "changed": {"paths": {"/a": {}}}, "removed": {"paths": {}}}
        fresh = {"kept": {"paths": {}}, "changed": {"paths": {"/b": {}}}, "added": {"paths": {}}}

        assert drifted_features(committed, fresh) == ("added", "changed", "removed")

    def test_identical_snapshots_have_no_drift(self):
        fragments = {"a": {"paths": {"/a": {}}, "components": {"schemas": {}}}}

        assert drifted_features(fragments, dict(fragments)) == ()


class TestRegisterLazyFeature:
    def test_skips_a_feature_that_was_already_loaded_before_registration_began(self):
        feature = SimpleNamespace(name="ok", module_path="json", register_fn=lambda app, module: None)

        assert _register_lazy_feature(SimpleNamespace(), feature, frozenset({"json"})) is None

    def test_registers_a_module_that_imports_cleanly(self):
        registered = []
        feature = SimpleNamespace(
            name="ok", module_path="json", register_fn=lambda app, module: registered.append(module.__name__)
        )

        assert _register_lazy_feature(SimpleNamespace(), feature, frozenset()) is None
        assert registered == ["json"]

    def test_registers_a_module_a_sibling_feature_dragged_into_sys_modules(self):
        """Regression: importing vector_stores imports vector_store_management as a side effect.

        Gating on live sys.modules treated that as "already registered", so its router was
        never included, it contributed no routes, and generate_snapshot dropped its whole
        fragment; POST /vector_store/new and friends disappeared from the OpenAPI spec.
        """
        registered = []
        feature = SimpleNamespace(
            name="sibling", module_path="json", register_fn=lambda app, module: registered.append(module.__name__)
        )

        assert "json" in sys.modules  # imported, but nobody registered its routes
        assert _register_lazy_feature(SimpleNamespace(), feature, preloaded=frozenset()) is None
        assert registered == ["json"]

    def test_reports_the_feature_and_error_when_the_module_is_missing(self):
        feature = SimpleNamespace(
            name="broken", module_path="litellm_no_such_module", register_fn=lambda app, module: None
        )

        result = _register_lazy_feature(SimpleNamespace(), feature, frozenset())

        assert result is not None
        name, error = result
        assert name == "broken"
        assert "ModuleNotFoundError" in error

    def test_reports_the_error_when_registration_raises(self):
        def _raise(app, module):
            raise ValueError("router blew up")

        feature = SimpleNamespace(name="broken", module_path="json", register_fn=_raise)

        assert _register_lazy_feature(SimpleNamespace(), feature, frozenset()) == (
            "broken",
            "ValueError: router blew up",
        )


class TestCheckSnapshot:
    """The check exists to fail when the committed file drifts; prove it does."""

    def test_passes_when_the_committed_file_matches(self, monkeypatch):
        fragments = {"guardrails": {"paths": {"/g": {}}, "components": {"schemas": {}}}}
        monkeypatch.setattr(snapshot_module, "generate_snapshot", lambda *, strict=False: fragments)
        monkeypatch.setattr(snapshot_module, "load_snapshot", lambda: dict(fragments))

        assert check_snapshot() == EXIT_IN_SYNC

    def test_fails_when_a_fragment_drifted(self, monkeypatch, capsys):
        monkeypatch.setattr(
            snapshot_module,
            "generate_snapshot",
            lambda *, strict=False: {"guardrails": {"paths": {"/new": {}}, "components": {"schemas": {}}}},
        )
        monkeypatch.setattr(
            snapshot_module,
            "load_snapshot",
            lambda: {"guardrails": {"paths": {"/old": {}}, "components": {"schemas": {}}}},
        )

        assert check_snapshot() == EXIT_DRIFTED
        assert "guardrails" in capsys.readouterr().err

    def test_fails_when_the_file_is_absent(self, monkeypatch):
        monkeypatch.setattr(snapshot_module, "generate_snapshot", lambda *, strict=False: {"a": {"paths": {}}})
        monkeypatch.setattr(snapshot_module, "load_snapshot", lambda: None)

        assert check_snapshot() == EXIT_DRIFTED

    def test_an_unimportable_feature_is_not_reported_as_drift(self, monkeypatch, capsys):
        """A skipped feature looks exactly like a deleted fragment, so it gets its own outcome.

        Reporting it as drift would tell you to regenerate a snapshot that is fine, and the
        regen would then silently drop the missing feature's fragments.
        """

        def _boom(*, strict: bool = False):
            raise RuntimeError("cannot verify the snapshot: mcp_app (ImportError: no module)")

        monkeypatch.setattr(snapshot_module, "generate_snapshot", _boom)

        assert check_snapshot() == EXIT_CANNOT_VERIFY
        assert check_snapshot() != EXIT_DRIFTED

        stderr = capsys.readouterr().err
        assert "mcp_app" in stderr
        assert "not snapshot drift" in stderr
        assert REGEN_COMMAND not in stderr

    def test_main_reports_the_import_failure_instead_of_raising(self, monkeypatch, capsys):
        """The workflow branches on the exit code, so this must not escape as a traceback."""

        def _boom(*, strict: bool = False):
            raise RuntimeError("cannot verify the snapshot: mcp_app (ImportError: no module)")

        monkeypatch.setattr(snapshot_module, "generate_snapshot", _boom)

        assert snapshot_module._main(["--check"]) == EXIT_CANNOT_VERIFY
        assert "mcp_app" in capsys.readouterr().err


class TestSerializeSnapshot:
    def test_is_stable_regardless_of_key_order(self):
        assert serialize_snapshot({"b": {"x": 1}, "a": {"y": 2}}) == serialize_snapshot({"a": {"y": 2}, "b": {"x": 1}})

    def test_ends_with_a_trailing_newline(self):
        assert serialize_snapshot({"a": {}}).endswith("}\n")
