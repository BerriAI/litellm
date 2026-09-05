import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Final

import pytest

REPO_ROOT: Final = Path(__file__).resolve().parents[2]
SCRIPT_PATH: Final = REPO_ROOT / "scripts" / "sync_cost_map.py"
FIXTURES: Final = Path(__file__).parent / "fixtures" / "cost_map_sync"
OPENROUTER_RAW: Final = (FIXTURES / "openrouter_models.json").read_bytes()
VERCEL_RAW: Final = (FIXTURES / "vercel_models.json").read_bytes()
NOW_MS: Final = 1757030400000

EXISTING_DEEPSEEK: Final = {
    "input_cost_per_token": 0.00000132,
    "input_cost_per_token_cache_hit": 4.4e-8,
    "litellm_provider": "openrouter",
    "max_input_tokens": 1048576,
    "max_output_tokens": 300000,
    "max_tokens": 300000,
    "mode": "chat",
    "output_cost_per_token": 0.00000396,
    "source": "https://openrouter.ai/deepseek/deepseek-v4-pro-0813",
    "supports_function_calling": True,
    "supports_prompt_caching": True,
    "supports_reasoning": True,
    "supports_response_schema": True,
    "supports_tool_choice": True,
}
EXISTING_GLM: Final = {
    "litellm_provider": "vercel_ai_gateway",
    "cache_read_input_token_cost": 1.1e-7,
    "input_cost_per_token": 4.5e-7,
    "max_input_tokens": 200000,
    "max_output_tokens": 200000,
    "max_tokens": 200000,
    "mode": "chat",
    "output_cost_per_token": 0.0000018,
    "source": "https://vercel.com/ai-gateway/models/glm-4.6",
    "supports_function_calling": True,
    "supports_parallel_function_calling": True,
    "supports_tool_choice": True,
}
BLOCK_END_OPENROUTER: Final = {
    "litellm_provider": "openrouter",
    "mode": "completion",
    "input_cost_per_token": 0.0000015,
    "output_cost_per_token": 0.000002,
}


def _base_map() -> dict[str, object]:
    return {
        "sample_spec": {"litellm_provider": "one of https://docs.litellm.ai/docs/providers"},
        "gpt-4o": {"litellm_provider": "openai", "mode": "chat"},
        "openrouter/deepseek/deepseek-v4-pro-0813": dict(EXISTING_DEEPSEEK),
        "openrouter/openai/gpt-3.5-turbo-instruct": dict(BLOCK_END_OPENROUTER),
        "vercel_ai_gateway/zai/glm-4.6": dict(EXISTING_GLM),
        "zzz/last": {"litellm_provider": "zzz", "mode": "chat"},
    }


@pytest.fixture(scope="module")
def sync() -> ModuleType:
    spec: Final = importlib.util.spec_from_file_location("sync_cost_map", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module: Final = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _openrouter_rows(*rows: dict[str, object]) -> bytes:
    return json.dumps(
        {"data": [{"pricing": {"prompt": "0.000001", "completion": "0.000002"}, **row} for row in rows]}
    ).encode()


def _vercel_rows(*rows: dict[str, object]) -> bytes:
    defaults: Final = {"type": "language", "pricing": {"input": "0.000001", "output": "0.000002"}}
    return json.dumps({"data": [{**defaults, **row} for row in rows]}).encode()


def _run(sync: ModuleType, cost_map: dict[str, object]):
    return sync.compute_sync(
        cost_map, (sync.load_openrouter(OPENROUTER_RAW), sync.load_vercel(VERCEL_RAW, now_ms=NOW_MS))
    )


def test_new_openrouter_entry_carries_catalog_prices_limits_and_capabilities(sync: ModuleType) -> None:
    outcome: Final = _run(sync, _base_map())

    assert outcome.cost_map["openrouter/inception/mercury-2.5-preview"] == {
        "cache_read_input_token_cost": 4e-9,
        "input_cost_per_token": 4e-8,
        "litellm_provider": "openrouter",
        "max_input_tokens": 260000,
        "max_output_tokens": 65536,
        "max_tokens": 65536,
        "mode": "chat",
        "output_cost_per_token": 1.5e-7,
        "source": "https://openrouter.ai/inception/mercury-2.5-preview",
        "supports_function_calling": True,
        "supports_reasoning": True,
        "supports_response_schema": True,
        "supports_tool_choice": True,
    }


def test_input_modalities_become_capability_flags(sync: ModuleType) -> None:
    outcome: Final = _run(sync, _base_map())

    gpt5: Final = outcome.cost_map["openrouter/openai/gpt-5-mini"]
    gemma: Final = outcome.cost_map["openrouter/google/gemma-4-26b-a4b-it:free"]
    vercel_gpt5: Final = outcome.cost_map["vercel_ai_gateway/openai/gpt-5-mini"]
    assert (gpt5["supports_vision"], gpt5["supports_pdf_input"]) == (True, True)
    assert "supports_video_input" not in gpt5 and "supports_audio_input" not in gpt5
    assert "input_cost_per_image" not in gpt5 and "supports_prompt_caching" not in gpt5
    assert (gemma["supports_vision"], gemma["supports_video_input"]) == (True, True)
    assert (vercel_gpt5["supports_vision"], vercel_gpt5["supports_pdf_input"]) == (True, True)
    assert vercel_gpt5["cache_read_input_token_cost"] == 2.5e-8
    assert vercel_gpt5["source"] == "https://vercel.com/ai-gateway/models/gpt-5-mini"


def test_free_models_are_added_with_zero_prices(sync: ModuleType) -> None:
    outcome: Final = _run(sync, _base_map())

    free: Final = outcome.cost_map["openrouter/cohere/north-mini-code:free"]
    assert (free["input_cost_per_token"], free["output_cost_per_token"]) == (0.0, 0.0)
    assert (free["max_input_tokens"], free["max_output_tokens"], free["max_tokens"]) == (256000, 64000, 64000)
    assert "cache_read_input_token_cost" not in free


def test_router_rows_and_unpriced_rows_are_skipped(sync: ModuleType) -> None:
    outcome: Final = _run(sync, _base_map())

    openrouter, vercel = outcome.providers
    assert "openrouter/openrouter/auto" not in outcome.cost_map
    assert "vercel_ai_gateway/perplexity/sonar" not in outcome.cost_map
    assert dict(openrouter.skipped) == {"unpriced or router": 1}
    assert dict(vercel.skipped) == {"deprecated": 1, "not token priced": 1, "no usable price": 1}


def test_vercel_rows_map_type_to_mode_and_drop_non_token_types(sync: ModuleType) -> None:
    outcome: Final = _run(sync, _base_map())

    embedding: Final = outcome.cost_map["vercel_ai_gateway/alibaba/qwen3-embedding-0.6b"]
    assert embedding == {
        "input_cost_per_token": 1e-8,
        "litellm_provider": "vercel_ai_gateway",
        "max_input_tokens": 32768,
        "max_output_tokens": 32768,
        "max_tokens": 32768,
        "mode": "embedding",
        "output_cost_per_token": 0.0,
        "source": "https://vercel.com/ai-gateway/models/qwen3-embedding-0.6b",
    }
    assert "vercel_ai_gateway/bfl/flux-2-flex" not in outcome.cost_map
    assert "vercel_ai_gateway/openai/gpt-4o-mini-transcribe" not in outcome.cost_map


def test_existing_entry_is_repriced_without_losing_curated_fields(sync: ModuleType) -> None:
    outcome: Final = _run(sync, _base_map())

    deepseek: Final = outcome.cost_map["openrouter/deepseek/deepseek-v4-pro-0813"]
    assert deepseek["input_cost_per_token"] == 5.7948e-7
    assert deepseek["output_cost_per_token"] == 1.73844e-6
    assert deepseek["cache_read_input_token_cost"] == 1.9316e-8
    assert deepseek["input_cost_per_token_cache_hit"] == 4.4e-8
    assert (deepseek["max_output_tokens"], deepseek["max_tokens"]) == (384000, 384000)
    glm: Final = outcome.cost_map["vercel_ai_gateway/zai/glm-4.6"]
    assert (glm["input_cost_per_token"], glm["output_cost_per_token"]) == (6e-7, 2.2e-6)
    assert glm["supports_parallel_function_calling"] is True
    assert glm["supports_reasoning"] is True
    assert (glm["max_output_tokens"], glm["max_tokens"]) == (200000, 200000)
    openrouter, vercel = outcome.providers
    assert [line.split(":")[0] for line in openrouter.updated] == ["openrouter/deepseek/deepseek-v4-pro-0813"]
    assert "input_cost_per_token: 1.32e-06 -> 5.7948e-07" in openrouter.updated[0]
    assert "max_output_tokens: 300000 -> 384000; max_tokens: 300000 -> 384000" in openrouter.updated[0]
    assert [line.split(":")[0] for line in vercel.updated] == ["vercel_ai_gateway/zai/glm-4.6"]
    assert vercel.warnings == (
        "vercel_ai_gateway/zai/glm-4.6: max_output_tokens: 200000 -> 96000 held back: a shrinking limit; "
        "max_tokens: 200000 -> 96000 held back: a shrinking limit",
    )


def test_legacy_max_tokens_moves_in_step_with_the_catalog_output_ceiling(sync: ModuleType) -> None:
    legacy: Final = {
        "input_cost_per_token": 4e-8,
        "litellm_provider": "openrouter",
        "max_tokens": 8192,
        "mode": "chat",
        "output_cost_per_token": 1.5e-7,
    }
    outcome: Final = _run(sync, {**_base_map(), "openrouter/inception/mercury-2.5-preview": legacy})

    mercury: Final = outcome.cost_map["openrouter/inception/mercury-2.5-preview"]
    assert (mercury["max_output_tokens"], mercury["max_tokens"]) == (65536, 65536)
    assert mercury["max_input_tokens"] == 260000
    assert list(mercury) == [
        *legacy,
        "cache_read_input_token_cost",
        "max_input_tokens",
        "max_output_tokens",
        "supports_function_calling",
        "supports_reasoning",
        "supports_response_schema",
        "supports_tool_choice",
    ]


def test_output_limits_stay_put_when_the_catalog_has_no_output_ceiling(sync: ModuleType) -> None:
    existing: Final = {
        "litellm_provider": "openrouter",
        "mode": "chat",
        "input_cost_per_token": 1e-6,
        "output_cost_per_token": 2e-6,
        "max_input_tokens": 1000,
        "max_output_tokens": 500,
        "max_tokens": 500,
    }
    catalog: Final = sync.load_openrouter(_openrouter_rows({"id": "acme/x", "context_length": 4000}))

    outcome: Final = sync.compute_sync({"openrouter/acme/x": dict(existing)}, (catalog,))

    entry: Final = outcome.cost_map["openrouter/acme/x"]
    assert (entry["max_input_tokens"], entry["max_output_tokens"], entry["max_tokens"]) == (4000, 500, 500)
    assert outcome.providers[0].updated == ("openrouter/acme/x: max_input_tokens: 1000 -> 4000",)
    assert outcome.providers[0].warnings == ()


def test_untouched_entries_survive_byte_for_byte(sync: ModuleType) -> None:
    outcome: Final = _run(sync, _base_map())

    assert outcome.cost_map["gpt-4o"] == {"litellm_provider": "openai", "mode": "chat"}
    assert outcome.cost_map["openrouter/openai/gpt-3.5-turbo-instruct"] == BLOCK_END_OPENROUTER
    assert outcome.cost_map["sample_spec"] == _base_map()["sample_spec"]


def test_second_sync_is_a_no_op(sync: ModuleType) -> None:
    first: Final = _run(sync, _base_map())

    second: Final = _run(sync, dict(first.cost_map))

    assert second.has_changes is False
    assert all(not provider.added and not provider.updated for provider in second.providers)
    assert list(second.cost_map) == list(first.cost_map)


def test_mode_mismatch_warns_and_leaves_the_entry_alone(sync: ModuleType) -> None:
    outcome: Final = _run(
        sync,
        {
            **_base_map(),
            "vercel_ai_gateway/openai/gpt-5-mini": {
                "litellm_provider": "vercel_ai_gateway",
                "mode": "responses",
                "input_cost_per_token": 1.0,
            },
        },
    )

    assert outcome.cost_map["vercel_ai_gateway/openai/gpt-5-mini"]["input_cost_per_token"] == 1.0
    vercel: Final = outcome.providers[1]
    assert "vercel_ai_gateway/openai/gpt-5-mini" not in vercel.added
    assert all("gpt-5-mini" not in line for line in vercel.updated)
    mismatch: Final = [line for line in vercel.warnings if "gpt-5-mini" in line]
    assert len(mismatch) == 1
    assert "vercel_ai_gateway/openai/gpt-5-mini" in mismatch[0]
    assert "'responses'" in mismatch[0] and "'chat'" in mismatch[0]


def test_new_keys_land_at_the_end_of_their_provider_block(sync: ModuleType) -> None:
    outcome: Final = _run(sync, _base_map())

    assert list(outcome.cost_map) == [
        "sample_spec",
        "gpt-4o",
        "openrouter/deepseek/deepseek-v4-pro-0813",
        "openrouter/openai/gpt-3.5-turbo-instruct",
        "openrouter/cohere/north-mini-code:free",
        "openrouter/google/gemma-4-26b-a4b-it:free",
        "openrouter/inception/mercury-2.5-preview",
        "openrouter/openai/gpt-5-mini",
        "vercel_ai_gateway/zai/glm-4.6",
        "vercel_ai_gateway/alibaba/qwen3-embedding-0.6b",
        "vercel_ai_gateway/openai/gpt-5-mini",
        "zzz/last",
    ]


def test_provider_without_a_block_is_appended_at_the_end(sync: ModuleType) -> None:
    outcome: Final = _run(sync, {"gpt-4o": {"litellm_provider": "openai", "mode": "chat"}})

    keys: Final = list(outcome.cost_map)
    assert keys[0] == "gpt-4o"
    assert keys[1:6] == sorted(keys[1:6]) and all(key.startswith("openrouter/") for key in keys[1:6])
    assert keys[6:] == sorted(keys[6:]) and all(key.startswith("vercel_ai_gateway/") for key in keys[6:])
    assert len(keys) == 9


def test_pr_body_lists_changes_per_provider(sync: ModuleType) -> None:
    body: Final = sync.render_pr_body(_run(sync, _base_map()))

    assert "## openrouter" in body and "## vercel_ai_gateway" in body
    assert "### Added (4)" in body and "- `openrouter/inception/mercury-2.5-preview`" in body
    assert "### Added (2)" in body and "- `vercel_ai_gateway/openai/gpt-5-mini`" in body
    assert "- `openrouter/deepseek/deepseek-v4-pro-0813: input_cost_per_token: 1.32e-06 -> 5.7948e-07" in body
    assert "Catalog rows skipped: deprecated (1), no usable price (1), not token priced (1)" in body


def test_pr_body_caps_every_section_so_a_large_first_sync_fits_github_limit(sync: ModuleType) -> None:
    lines: Final = tuple(f"provider/model-{index}: {'x' * 200}" for index in range(400))
    outcome: Final = sync.SyncOutcome(
        cost_map={},
        providers=tuple(
            sync.ProviderOutcome(provider=provider, added=lines, updated=lines, warnings=lines, skipped={})
            for provider in ("openrouter", "vercel_ai_gateway")
        ),
    )

    body: Final = sync.render_pr_body(outcome)

    assert len(body) < 65_536
    assert body.count("### Added (400)") == 2 and body.count("- and 370 more, see the diff") == 4
    assert body.count("- and 370 more, see the workflow log") == 2
    assert body.count("- `provider/model-29: ") == 4 and "provider/model-30: " not in body


def test_workflow_log_lists_every_warning_the_capped_pr_body_drops(sync: ModuleType, tmp_path: Path, capsys) -> None:
    keys: Final = tuple(f"openrouter/acme/model-{index:02d}" for index in range(40))
    catalog: Final = {
        "data": [
            {"id": key.removeprefix("openrouter/"), "pricing": {"prompt": "0.000001", "completion": "0.000002"}}
            for key in keys
        ]
    }
    cost_map: Final = {**_base_map(), **{key: {"litellm_provider": "openrouter", "mode": "embedding"} for key in keys}}
    for relpath in ("model_prices_and_context_window.json", "litellm/model_prices_and_context_window_backup.json"):
        (tmp_path / relpath).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / relpath).write_text(json.dumps(cost_map, indent=4) + "\n")
    (tmp_path / "openrouter.json").write_text(json.dumps(catalog))
    body_file: Final = tmp_path / "body.md"

    code: Final = sync.main(
        [
            "--openrouter-json",
            str(tmp_path / "openrouter.json"),
            "--vercel-json",
            str(FIXTURES / "vercel_models.json"),
            "--pr-body-file",
            str(body_file),
            "--repo-root",
            str(tmp_path),
        ]
    )

    body: Final = body_file.read_text()
    log: Final = capsys.readouterr().out
    assert code == 0
    assert "### Warnings needing a human call (40)" in body and "- and 10 more, see the workflow log" in body
    assert keys[29] in body and keys[30] not in body
    assert all(key in log for key in keys) and "more, see the" not in log


@pytest.mark.parametrize(
    ("loader", "raw"),
    [
        ("load_openrouter", b'{"data": []}'),
        ("load_openrouter", b'{"data": [{"id": "x", "pricing": {"prompt": 1}}]}'),
        ("load_vercel", b"[]"),
        ("load_vercel", b'{"data": [{"id": "x"}]}'),
    ],
)
def test_malformed_catalogs_fail_the_run(sync: ModuleType, loader: str, raw: bytes) -> None:
    kwargs: Final = {"now_ms": NOW_MS} if loader == "load_vercel" else {}
    with pytest.raises(sync.SyncError):
        getattr(sync, loader)(raw, **kwargs)


@pytest.mark.parametrize("price", ["Infinity", "-Infinity", "NaN"])
def test_non_finite_catalog_prices_count_as_unpriced(sync: ModuleType, price: str) -> None:
    raw: Final = json.dumps({"data": [{"id": "acme/x", "pricing": {"prompt": price, "completion": "0"}}]}).encode()

    catalog: Final = sync.load_openrouter(raw)

    assert catalog.entries == ()
    assert dict(catalog.skipped) == {"unpriced or router": 1}


def _vercel_language_row(deprecated_at: int | None) -> bytes:
    row: Final = {
        "id": "acme/chat-1",
        "type": "language",
        "context_window": 1000,
        "max_tokens": 100,
        "pricing": {"input": "0.000001", "output": "0.000002"},
        "deprecated_at": deprecated_at,
    }
    return json.dumps({"data": [row]}).encode()


def test_a_scheduled_deprecation_keeps_syncing_until_the_date(sync: ModuleType) -> None:
    scheduled: Final = sync.load_vercel(_vercel_language_row(NOW_MS + 1), now_ms=NOW_MS)
    passed: Final = sync.load_vercel(_vercel_language_row(NOW_MS), now_ms=NOW_MS)

    assert [entry.key for entry in scheduled.entries] == ["vercel_ai_gateway/acme/chat-1"]
    assert dict(scheduled.skipped).get("deprecated", 0) == 0
    assert passed.entries == ()
    assert dict(passed.skipped)["deprecated"] == 1


def _repo(tmp_path: Path) -> Path:
    for relpath in ("model_prices_and_context_window.json", "litellm/model_prices_and_context_window_backup.json"):
        target = tmp_path / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(_base_map(), indent=4) + "\n")
    return tmp_path


def test_write_updates_both_cost_map_files_identically(sync: ModuleType, tmp_path: Path, capsys) -> None:
    repo: Final = _repo(tmp_path)
    body_file: Final = tmp_path / "body.md"

    code: Final = sync.main(
        [
            "--write",
            "--openrouter-json",
            str(FIXTURES / "openrouter_models.json"),
            "--vercel-json",
            str(FIXTURES / "vercel_models.json"),
            "--pr-body-file",
            str(body_file),
            "--repo-root",
            str(repo),
        ]
    )

    root: Final = (repo / "model_prices_and_context_window.json").read_text()
    backup: Final = (repo / "litellm" / "model_prices_and_context_window_backup.json").read_text()
    assert code == 0
    assert root == backup
    assert root.endswith("}\n")
    assert json.loads(root)["openrouter/inception/mercury-2.5-preview"]["input_cost_per_token"] == 4e-8
    assert "### Added (4)" in body_file.read_text()
    assert capsys.readouterr().out.startswith("openrouter: added=4 updated=1 warnings=0")


def test_dry_run_touches_nothing(sync: ModuleType, tmp_path: Path, capsys) -> None:
    repo: Final = _repo(tmp_path)
    before: Final = (repo / "model_prices_and_context_window.json").read_bytes()

    code: Final = sync.main(
        [
            "--openrouter-json",
            str(FIXTURES / "openrouter_models.json"),
            "--vercel-json",
            str(FIXTURES / "vercel_models.json"),
            "--repo-root",
            str(repo),
        ]
    )

    assert code == 0
    assert (repo / "model_prices_and_context_window.json").read_bytes() == before
    assert "dry run: no files were touched" in capsys.readouterr().out


def test_new_entries_inherit_model_intrinsic_traits_from_the_root_entry(sync: ModuleType) -> None:
    root: Final = {
        "litellm_provider": "anthropic",
        "mode": "chat",
        "input_cost_per_token": 5e-6,
        "supports_adaptive_thinking": True,
        "thinking_always_on": True,
        "supports_sampling_params": False,
        "supports_function_calling": False,
        "supports_vision": True,
        "prompt_cache_min_tokens": 1024,
        "supports_web_search": True,
    }
    already_synced: Final = {"litellm_provider": "openrouter", "mode": "chat", "input_cost_per_token": 5e-6}
    cost_map: Final = {
        "claude-fable-5": dict(root),
        "claude-fable-5-1": {**root, "prompt_cache_min_tokens": 512},
        "claude-embed-5": {**root, "mode": "embedding"},
        "openrouter/anthropic/claude-fable-5:thinking": dict(already_synced),
    }
    openrouter: Final = sync.load_openrouter(
        _openrouter_rows(
            {"id": "anthropic/claude-fable-5:batch", "supported_parameters": ["tools"]},
            {"id": "anthropic/claude-fable-5:thinking"},
            {"id": "anthropic/claude-embed-5"},
            {"id": "anthropic/claude-opus-6"},
        )
    )
    vercel: Final = sync.load_vercel(
        _vercel_rows({"id": "anthropic/claude-fable-5.1"}, {"id": "anthropic/claude-fable-5.1-fast"}), now_ms=NOW_MS
    )

    outcome: Final = sync.compute_sync(cost_map, (openrouter, vercel))

    batch: Final = outcome.cost_map["openrouter/anthropic/claude-fable-5:batch"]
    assert (batch["supports_adaptive_thinking"], batch["thinking_always_on"]) == (True, True)
    assert (batch["supports_sampling_params"], batch["prompt_cache_min_tokens"]) == (False, 1024)
    assert batch["supports_function_calling"] is True
    assert not {"supports_web_search", "supports_vision"} & batch.keys()
    assert outcome.cost_map["vercel_ai_gateway/anthropic/claude-fable-5.1"]["prompt_cache_min_tokens"] == 512
    fast: Final = outcome.cost_map["vercel_ai_gateway/anthropic/claude-fable-5.1-fast"]
    assert (fast["prompt_cache_min_tokens"], fast["supports_adaptive_thinking"]) == (512, True)
    assert "prompt_cache_min_tokens" not in outcome.cost_map["openrouter/anthropic/claude-opus-6"]
    assert "supports_adaptive_thinking" not in outcome.cost_map["openrouter/anthropic/claude-embed-5"]
    assert "supports_adaptive_thinking" not in outcome.cost_map["openrouter/anthropic/claude-fable-5:thinking"]


@pytest.mark.parametrize(
    ("field", "old", "new", "reason"),
    [
        ("input_cost_per_token", 1e-6, 0.0, "a price crossing zero"),
        ("input_cost_per_token", 0.0, 1e-6, "a price crossing zero"),
        ("output_cost_per_token", 1e-7, 2e-6, "a price moving more than 10x"),
        ("output_cost_per_token", 2e-6, 1e-7, "a price moving more than 10x"),
        ("max_input_tokens", 200000, 128000, "a shrinking limit"),
    ],
)
def test_out_of_bounds_changes_are_held_back_as_warnings(
    sync: ModuleType, field: str, old: float, new: float, reason: str
) -> None:
    existing: Final = {
        "litellm_provider": "openrouter",
        "mode": "chat",
        "input_cost_per_token": 1e-6,
        "output_cost_per_token": 2e-6,
        "max_input_tokens": 200000,
        field: old,
    }
    catalog_row: Final = {
        "id": "acme/x",
        "context_length": 200000,
        "pricing": {"prompt": "0.000001", "completion": "0.000002"},
        **({"context_length": int(new)} if field == "max_input_tokens" else {}),
    }
    catalog_row["pricing"] = {
        **catalog_row["pricing"],
        **({"prompt": str(new)} if field == "input_cost_per_token" else {}),
        **({"completion": str(new)} if field == "output_cost_per_token" else {}),
    }

    outcome: Final = sync.compute_sync(
        {"openrouter/acme/x": dict(existing)}, (sync.load_openrouter(_openrouter_rows(catalog_row)),)
    )

    assert outcome.cost_map["openrouter/acme/x"] == existing
    assert outcome.has_changes is False
    assert outcome.providers[0].warnings == (f"openrouter/acme/x: {field}: {old!r} -> {new!r} held back: {reason}",)


def test_a_price_move_within_ten_x_is_applied(sync: ModuleType) -> None:
    existing: Final = {"litellm_provider": "openrouter", "mode": "chat", "input_cost_per_token": 1e-6}
    catalog: Final = sync.load_openrouter(
        _openrouter_rows({"id": "acme/x", "pricing": {"prompt": "0.000009", "completion": "0"}})
    )

    outcome: Final = sync.compute_sync({"openrouter/acme/x": existing}, (catalog,))

    assert outcome.cost_map["openrouter/acme/x"]["input_cost_per_token"] == 9e-6
    assert outcome.providers[0].warnings == ()


def test_vercel_long_context_tiers_map_to_above_threshold_prices(sync: ModuleType) -> None:
    pricing: Final = {
        "input": "0.0000015",
        "input_tiers": [
            {"cost": "0.0000015", "min": 0, "max": 128001},
            {"cost": "0.0000027", "min": 128001, "max": 200001},
            {"cost": "0.0000045", "min": 200001},
        ],
        "output": "0.0000075",
        "output_tiers": [{"cost": "0.0000075", "max": 200001}, {"cost": "0.00001125", "min": 200001}],
        "input_cache_read": "0.0000003",
        "input_cache_read_tiers": [
            {"cost": "0.0000003", "min": 0, "max": 200001},
            {"cost": "0.0000006", "min": 200001},
        ],
        "input_cache_write": "0.000002",
        "input_cache_write_tiers": [{"cost": "0.000002", "min": 0, "max": 200001}, {"cost": "0.000004", "min": 200001}],
    }
    catalog: Final = sync.load_vercel(_vercel_rows({"id": "acme/long", "pricing": pricing}), now_ms=NOW_MS)

    outcome: Final = sync.compute_sync({}, (catalog,))

    entry: Final = outcome.cost_map["vercel_ai_gateway/acme/long"]
    assert entry["input_cost_per_token"] == 1.5e-6
    assert entry["input_cost_per_token_above_128k_tokens"] == 2.7e-6
    assert entry["input_cost_per_token_above_200k_tokens"] == 4.5e-6
    assert entry["output_cost_per_token_above_200k_tokens"] == 1.125e-5
    assert entry["cache_read_input_token_cost_above_200k_tokens"] == 6e-7
    assert entry["cache_creation_input_token_cost_above_200k_tokens"] == 4e-6
    assert not any(key.endswith("_above_0k_tokens") for key in entry)


def test_output_or_cache_tiers_without_matching_input_tier_skip_the_row(sync: ModuleType) -> None:
    pricing: Final = {
        "input": "0.0000015",
        "input_tiers": [
            {"cost": "0.0000015", "min": 0, "max": 32001},
            {"cost": "0.0000027", "min": 32001, "max": 128001},
            {"cost": "0.0000045", "min": 128001},
        ],
        "output": "0.0000075",
        "output_tiers": [{"cost": "0.0000075", "max": 200001}, {"cost": "0.00001125", "min": 200001}],
    }
    catalog: Final = sync.load_vercel(_vercel_rows({"id": "acme/orphan", "pricing": pricing}), now_ms=NOW_MS)

    outcome: Final = sync.compute_sync({}, (catalog,))

    assert "vercel_ai_gateway/acme/orphan" not in outcome.cost_map
    assert dict(catalog.skipped) == {"tiers outside the registry's thresholds": 1}
    assert outcome.providers[0].warnings == (
        "vercel_ai_gateway/acme/orphan: output_cost_per_token tier at 200k_tokens has no matching input tier; "
        "row skipped",
    )


@pytest.mark.parametrize(
    ("tiers", "problem"),
    [
        (
            [{"cost": "0.000001", "min": 0, "max": 150500}, {"cost": "0.000002", "min": 150500}],
            "input_cost_per_token tiers have a boundary that is not a whole thousand or an unusable price",
        ),
        (
            [{"cost": "0.000001", "min": 0, "max": 128000}, {"cost": "0.000002", "min": 200000}],
            "input_cost_per_token tiers are not contiguous",
        ),
        (
            [{"cost": "0.000001", "min": 0, "max": 128000}, {"cost": "0.000002", "min": 128000, "max": 256000}],
            "input_cost_per_token tiers are not contiguous",
        ),
    ],
)
def test_unmappable_tiers_skip_the_row_with_a_warning(sync: ModuleType, tiers: list, problem: str) -> None:
    pricing: Final = {"input": "0.000001", "input_tiers": tiers, "output": "0.000002"}
    catalog: Final = sync.load_vercel(_vercel_rows({"id": "acme/odd", "pricing": pricing}), now_ms=NOW_MS)

    outcome: Final = sync.compute_sync({}, (catalog,))

    assert "vercel_ai_gateway/acme/odd" not in outcome.cost_map
    assert dict(catalog.skipped) == {"tiers outside the registry's thresholds": 1}
    assert outcome.providers[0].warnings == (f"vercel_ai_gateway/acme/odd: {problem}; row skipped",)


def test_a_price_that_varies_by_provider_seeds_but_never_overwrites(sync: ModuleType) -> None:
    curated: Final = {
        "litellm_provider": "vercel_ai_gateway",
        "mode": "chat",
        "input_cost_per_token": 9e-7,
        "output_cost_per_token": 2e-6,
        "max_input_tokens": 100000,
    }
    row: Final = {
        "context_window": 262144,
        "pricing": {
            "input": "0.0000015",
            "input_tiers": [{"cost": "0.0000015", "min": 0, "max": 128001}, {"cost": "0.000003", "min": 128001}],
            "output": "0.000002",
            "input_cache_read": "0.0000003",
            "varies_by_provider": True,
        },
    }
    catalog: Final = sync.load_vercel(
        _vercel_rows({"id": "acme/curated", **row}, {"id": "acme/fresh", **row}), now_ms=NOW_MS
    )

    outcome: Final = sync.compute_sync({"vercel_ai_gateway/acme/curated": dict(curated)}, (catalog,))

    existing: Final = outcome.cost_map["vercel_ai_gateway/acme/curated"]
    assert (existing["input_cost_per_token"], existing["max_input_tokens"]) == (9e-7, 262144)
    assert not any("cache_read" in name or "_above_" in name for name in existing)
    fresh: Final = outcome.cost_map["vercel_ai_gateway/acme/fresh"]
    assert (fresh["input_cost_per_token"], fresh["input_cost_per_token_above_128k_tokens"]) == (1.5e-6, 3e-6)
    assert fresh["cache_read_input_token_cost"] == 3e-7
    assert outcome.providers[0].warnings == (
        "vercel_ai_gateway/acme/curated: input_cost_per_token: 9e-07 -> 1.5e-06 held back: "
        "the catalog price varies by provider; cache_read_input_token_cost: None -> 3e-07 held back: "
        "the catalog price varies by provider; input_cost_per_token_above_128k_tokens: None -> 3e-06 held back: "
        "the catalog price varies by provider",
    )


def test_image_and_audio_outputs_are_priced_per_token_or_skipped(sync: ModuleType) -> None:
    openrouter: Final = sync.load_openrouter(
        _openrouter_rows(
            {
                "id": "openai/gpt-5-image",
                "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["image", "text"]},
                "pricing": {"prompt": "0.00001", "completion": "0.00001", "image_output": "0.00004"},
            },
            {
                "id": "acme/talker",
                "architecture": {"input_modalities": ["text", "audio"], "output_modalities": ["text", "audio"]},
                "pricing": {
                    "prompt": "0.000001",
                    "completion": "0.000002",
                    "audio": "0.000005",
                    "audio_output": "0.00001",
                },
            },
            {
                "id": "acme/mute",
                "architecture": {"input_modalities": ["text"], "output_modalities": ["text", "audio"]},
            },
            {"id": "acme/painter", "architecture": {"output_modalities": ["image"]}},
        )
    )
    vercel: Final = sync.load_vercel(
        _vercel_rows(
            {
                "id": "acme/speaker",
                "modalities": {"input": ["text", "audio"], "output": ["text", "audio"]},
                "pricing": {
                    "input": "0.000001",
                    "output": "0.000002",
                    "audio_input_token_cost": "0.000004",
                    "audio_output_token_cost": "0.000008",
                },
            },
            {"id": "acme/drawer", "modalities": {"input": ["text"], "output": ["text", "image"]}},
        ),
        now_ms=NOW_MS,
    )

    outcome: Final = sync.compute_sync({}, (openrouter, vercel))

    image: Final = outcome.cost_map["openrouter/openai/gpt-5-image"]
    assert (image["output_cost_per_image_token"], image["mode"], image["supports_vision"]) == (4e-5, "chat", True)
    talker: Final = outcome.cost_map["openrouter/acme/talker"]
    assert (talker["input_cost_per_audio_token"], talker["output_cost_per_audio_token"]) == (5e-6, 1e-5)
    assert (talker["supports_audio_input"], talker["supports_audio_output"]) == (True, True)
    speaker: Final = outcome.cost_map["vercel_ai_gateway/acme/speaker"]
    assert (speaker["input_cost_per_audio_token"], speaker["output_cost_per_audio_token"]) == (4e-6, 8e-6)
    assert speaker["supports_audio_output"] is True
    assert {"openrouter/acme/mute", "openrouter/acme/painter", "vercel_ai_gateway/acme/drawer"}.isdisjoint(
        outcome.cost_map
    )
    assert dict(openrouter.skipped) == {"output priced outside the catalog": 2}
    assert dict(vercel.skipped) == {"output priced outside the catalog": 1}


def test_updates_keep_the_curated_key_order_and_append_new_keys(sync: ModuleType) -> None:
    curated: Final = {
        "mode": "chat",
        "output_cost_per_token": 2e-6,
        "input_cost_per_token": 1e-6,
        "litellm_provider": "openrouter",
    }
    catalog: Final = sync.load_openrouter(
        _openrouter_rows(
            {
                "id": "acme/x",
                "pricing": {"prompt": "0.000003", "completion": "0.000002", "input_cache_read": "0.0000001"},
            }
        )
    )

    outcome: Final = sync.compute_sync({"openrouter/acme/x": dict(curated)}, (catalog,))

    assert list(outcome.cost_map["openrouter/acme/x"]) == [*curated, "cache_read_input_token_cost"]
    assert outcome.cost_map["openrouter/acme/x"]["input_cost_per_token"] == 3e-6
