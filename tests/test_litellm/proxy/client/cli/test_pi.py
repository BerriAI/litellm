import json
import os
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import requests

from litellm.proxy.client.cli.commands.pi import (
    ListingFailure,
    ModelLimits,
    PiSyncError,
    fetch_model_ids,
    fetch_model_limits,
    fetch_model_listing,
    models_json_path,
    provider_block,
    sync_models_json,
)


class _FakeResponse:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


def _refused(*args, **kwargs):
    raise requests.ConnectionError("refused")


class TestFetchModelIds:
    def test_returns_ids_in_proxy_order_deduped(self):
        captured = {}

        def fake_get(url, headers, timeout):
            captured["url"] = url
            captured["headers"] = headers
            return _FakeResponse(
                200,
                {"data": [{"id": "m-b"}, {"id": "m-a"}, {"id": "m-b"}]},
            )

        assert fetch_model_ids("http://localhost:4000/", "sk-key", get=fake_get) == ("m-b", "m-a")
        assert captured["url"] == "http://localhost:4000/v1/models"
        assert captured["headers"] == {"Authorization": "Bearer sk-key"}

    def test_returns_rows_with_optional_source_model_and_dedups_identical_rows(self):
        result = fetch_model_listing(
            "http://localhost:4000",
            "sk-key",
            get=lambda *a, **k: _FakeResponse(
                200,
                {"data": [{"id": "emitted", "source_model": "source"}, {"id": "emitted", "source_model": "source"}]},
            ),
        )
        assert not isinstance(result, PiSyncError)
        assert tuple((model.id, model.source_model) for model in result) == (("emitted", "source"),)

    @pytest.mark.parametrize(
        "entry",
        [
            {"id": ""},
            {"id": "emitted", "source_model": ""},
            {"id": "emitted", "source_model": 1},
        ],
    )
    def test_rejects_invalid_model_identity(self, entry):
        result = fetch_model_listing(
            "http://localhost:4000", "sk-key", get=lambda *a, **k: _FakeResponse(200, {"data": [entry]})
        )
        assert isinstance(result, PiSyncError) and result.kind is ListingFailure.BAD_BODY

    def test_rejects_conflicting_emitted_id_mappings(self):
        result = fetch_model_listing(
            "http://localhost:4000",
            "sk-key",
            get=lambda *a, **k: _FakeResponse(
                200,
                {"data": [{"id": "emitted", "source_model": "one"}, {"id": "emitted", "source_model": "two"}]},
            ),
        )
        assert isinstance(result, PiSyncError) and result.kind is ListingFailure.BAD_BODY

    def test_network_error_is_a_value(self):
        def boom(*a, **k):
            raise requests.ConnectionError("refused")

        result = fetch_model_ids("http://localhost:4000", "sk-key", get=boom)
        assert isinstance(result, PiSyncError)
        assert "Could not list models" in result.message

    def test_non_200_is_a_value(self):
        result = fetch_model_ids("http://localhost:4000", "sk-key", get=lambda *a, **k: _FakeResponse(500))
        assert isinstance(result, PiSyncError)
        assert "HTTP 500" in result.message

    def test_malformed_body_is_a_value(self):
        result = fetch_model_ids(
            "http://localhost:4000",
            "sk-key",
            get=lambda *a, **k: _FakeResponse(200, {"data": "nope"}),
        )
        assert isinstance(result, PiSyncError)

    def test_empty_model_list_is_a_value(self):
        result = fetch_model_ids(
            "http://localhost:4000",
            "sk-key",
            get=lambda *a, **k: _FakeResponse(200, {"data": []}),
        )
        assert isinstance(result, PiSyncError)
        assert "no models" in result.message
        assert result.kind is ListingFailure.EMPTY

    @pytest.mark.parametrize(
        ("get", "kind"),
        [
            (_refused, ListingFailure.UNREACHABLE),
            (lambda *a, **k: _FakeResponse(401), ListingFailure.REJECTED),
            (lambda *a, **k: _FakeResponse(403), ListingFailure.REJECTED),
            (lambda *a, **k: _FakeResponse(500), ListingFailure.OTHER),
            (lambda *a, **k: _FakeResponse(200), ListingFailure.BAD_BODY),
        ],
        ids=["unreachable", "401", "403", "500", "bad-body"],
    )
    def test_the_failure_kind_is_decided_where_the_response_is_classified(self, get, kind):
        result = fetch_model_ids("http://localhost:4000", "sk-key", get=get)
        assert isinstance(result, PiSyncError) and result.kind is kind


class TestFetchModelLimits:
    def test_maps_group_limits_and_hits_model_group_info(self):
        captured = {}

        def fake_get(url, headers, timeout):
            captured["url"] = url
            return _FakeResponse(
                200,
                {
                    "data": [
                        {"model_group": "m-a", "max_input_tokens": 131072, "max_output_tokens": 8192},
                        {"model_group": "m-b", "max_input_tokens": None, "max_output_tokens": None},
                    ]
                },
            )

        limits = fetch_model_limits("http://localhost:4000/", "sk-key", get=fake_get)
        assert captured["url"] == "http://localhost:4000/model_group/info"
        assert limits["m-a"] == ModelLimits(context_window=131072, max_tokens=8192)
        assert limits["m-b"] == ModelLimits(context_window=None, max_tokens=None)

    def test_non_200_degrades_to_no_limits(self):
        assert fetch_model_limits("http://localhost:4000", "sk-key", get=lambda *a, **k: _FakeResponse(403)) == {}

    def test_network_error_degrades_to_no_limits(self):
        def boom(*a, **k):
            raise requests.ConnectionError("refused")

        assert fetch_model_limits("http://localhost:4000", "sk-key", get=boom) == {}

    def test_malformed_body_degrades_to_no_limits(self):
        assert (
            fetch_model_limits(
                "http://localhost:4000", "sk-key", get=lambda *a, **k: _FakeResponse(200, {"data": "nope"})
            )
            == {}
        )


class TestModelsJsonPath:
    def test_env_override_wins(self):
        assert models_json_path({"PI_CODING_AGENT_DIR": "/custom/dir"}) == Path("/custom/dir/models.json")

    def test_defaults_to_home_pi_agent(self):
        assert models_json_path({}) == Path.home() / ".pi" / "agent" / "models.json"


class TestProviderBlock:
    def test_points_pi_at_proxy_with_env_interpolated_key(self):
        block = provider_block("http://localhost:4000/", ("m-1", "m-2"))
        assert block == {
            "baseUrl": "http://localhost:4000/v1",
            "api": "openai-completions",
            "apiKey": "$LITELLM_PROXY_API_KEY",
            "models": [{"id": "m-1"}, {"id": "m-2"}],
        }

    def test_known_limits_become_context_window_and_max_tokens(self):
        block = provider_block(
            "http://localhost:4000",
            ("m-1", "m-2"),
            {
                "m-1": ModelLimits(context_window=131072, max_tokens=8192),
                "m-2": ModelLimits(context_window=None, max_tokens=None),
            },
        )
        assert block["models"] == [
            {"id": "m-1", "contextWindow": 131072, "maxTokens": 8192},
            {"id": "m-2"},
        ]


class TestSyncModelsJson:
    def test_creates_file_and_parent_dirs(self, tmp_path):
        path = tmp_path / "agent" / "models.json"
        assert sync_models_json(path, "http://localhost:4000", ("m-1",)) is None
        written = json.loads(path.read_text())
        assert written["providers"]["litellm"]["baseUrl"] == "http://localhost:4000/v1"
        assert written["providers"]["litellm"]["models"] == [{"id": "m-1"}]

    def test_preserves_other_providers_and_top_level_keys(self, tmp_path):
        path = tmp_path / "models.json"
        path.write_text(
            json.dumps(
                {
                    "somethingElse": True,
                    "providers": {
                        "ollama": {"baseUrl": "http://localhost:11434/v1"},
                        "litellm": {"baseUrl": "http://stale:1234/v1", "models": []},
                    },
                }
            )
        )
        assert sync_models_json(path, "http://localhost:4000", ("m-1",)) is None
        written = json.loads(path.read_text())
        assert written["somethingElse"] is True
        assert written["providers"]["ollama"] == {"baseUrl": "http://localhost:11434/v1"}
        assert written["providers"]["litellm"]["baseUrl"] == "http://localhost:4000/v1"
        assert written["providers"]["litellm"]["models"] == [{"id": "m-1"}]

    def test_write_leaves_no_staging_file_behind(self, tmp_path):
        path = tmp_path / "models.json"
        assert sync_models_json(path, "http://localhost:4000", ("m-1",)) is None
        assert [p.name for p in tmp_path.iterdir()] == ["models.json"]

    def test_written_file_is_private(self, tmp_path):
        path = tmp_path / "models.json"
        assert sync_models_json(path, "http://localhost:4000", ("m-1",)) is None
        if os.name != "nt":
            assert stat.S_IMODE(path.stat().st_mode) == 0o600

        path.write_text(json.dumps({"providers": {"other": {"apiKey": "literal-secret"}}}))
        path.chmod(0o644)
        assert sync_models_json(path, "http://localhost:4000", ("m-2",)) is None
        if os.name != "nt":
            assert stat.S_IMODE(path.stat().st_mode) == 0o600

    def test_concurrent_syncs_do_not_collide(self, tmp_path):
        path = tmp_path / "models.json"
        model_lists = (("m-a",), ("m-b",))

        def sync(model_ids):
            return sync_models_json(path, "http://localhost:4000", model_ids)

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = [result for _ in range(30) for result in executor.map(sync, model_lists)]

        assert results == [None] * 60
        written = json.loads(path.read_text())
        assert written["providers"]["litellm"]["models"] in ([{"id": "m-a"}], [{"id": "m-b"}])
        assert list(tmp_path.glob("models.json.*.tmp")) == []

    def test_invalid_json_is_a_value_and_file_untouched(self, tmp_path):
        path = tmp_path / "models.json"
        path.write_text("{not json")
        result = sync_models_json(path, "http://localhost:4000", ("m-1",))
        assert isinstance(result, PiSyncError)
        assert path.read_text() == "{not json"

    def test_non_object_providers_is_a_value(self, tmp_path):
        path = tmp_path / "models.json"
        path.write_text(json.dumps({"providers": ["nope"]}))
        result = sync_models_json(path, "http://localhost:4000", ("m-1",))
        assert isinstance(result, PiSyncError)

    def test_top_level_non_object_is_a_value(self, tmp_path):
        path = tmp_path / "models.json"
        path.write_text(json.dumps(["nope"]))
        result = sync_models_json(path, "http://localhost:4000", ("m-1",))
        assert isinstance(result, PiSyncError)

    def test_unwritable_path_is_a_value(self, tmp_path):
        blocker = tmp_path / "agent"
        blocker.write_text("i am a file, not a directory")
        result = sync_models_json(blocker / "models.json", "http://localhost:4000", ("m-1",))
        assert isinstance(result, PiSyncError)
        assert "Could not" in result.message
