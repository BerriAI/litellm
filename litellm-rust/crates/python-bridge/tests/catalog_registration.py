from __future__ import annotations

import inspect
import json
import unittest
from typing import Final
from unittest.mock import patch

from pydantic import JsonValue

import litellm
from litellm.rust_bridge import _native as native
from litellm.utils import _fallback_registration, _resolve_builtin_model_cost_entry, _without_overridden_price_variants


class RegistrationTargetTests(unittest.TestCase):
    def test_registration_aliases_match_legacy_candidate_order(self) -> None:
        rows: Final[dict[str, object]] = {
            "us.anthropic.model": {"litellm_provider": "bedrock", "input_cost_per_token": 0.001},
            "anthropic.model": {"litellm_provider": "bedrock_converse", "input_cost_per_token": 0.002},
            "anthropic.Model": {"litellm_provider": "bedrock_converse", "input_cost_per_token": 0.003},
            "bedrock/bedrock/us.anthropic.priority": {"litellm_provider": "bedrock"},
            "bedrock/us.anthropic.priority": {"litellm_provider": "bedrock"},
            "us.anthropic.priority": {"litellm_provider": "bedrock"},
            "anthropic.priority": {"litellm_provider": "bedrock"},
            "ap-northeast-1.anthropic.priority": {"litellm_provider": "bedrock"},
            "anthropic.dated-2026-01-01": {"litellm_provider": "bedrock"},
            "bedrock/anthropic.dated": {"litellm_provider": "bedrock"},
            "model": {"litellm_provider": "openai"},
            "openai/model": {"litellm_provider": "openai"},
            "hosted": {"litellm_provider": "hosted_vllm"},
            "anthropic.null": {"litellm_provider": None},
            "anthropic.absent": {},
        }
        cases: Final[tuple[tuple[str, str | None, str | None], ...]] = (
            ("bedrock/bedrock/bedrock/us.anthropic.model", "bedrock", "us.anthropic.model"),
            ("bedrock/bedrock/bedrock/us.anthropic.model", None, "us.anthropic.model"),
            ("bedrock/bedrock/bedrock/us.anthropic.model", "", "us.anthropic.model"),
            ("bedrock/bedrock/bedrock/bedrock/converse/us.anthropic.model", "bedrock", "us.anthropic.model"),
            ("bedrock/bedrock/bedrock/us.anthropic.priority", "bedrock", "bedrock/bedrock/us.anthropic.priority"),
            ("bedrock/ap-northeast-1.anthropic.model", "bedrock", "anthropic.model"),
            ("bedrock/converse/ap-northeast-1.anthropic.model", "bedrock", "anthropic.model"),
            ("bedrock/invoke/ap-northeast-1.anthropic.model", "bedrock", "anthropic.model"),
            ("bedrock/chat_completions/ap-northeast-1.anthropic.model", "bedrock", "anthropic.model"),
            ("bedrock/openai/ap-northeast-1.anthropic.model", "bedrock", "anthropic.model"),
            ("bedrock/mantle/ap-northeast-1.anthropic.model", "bedrock", "anthropic.model"),
            ("bedrock/nova/ap-northeast-1.anthropic.model", "bedrock", "anthropic.model"),
            ("bedrock/nova-2/ap-northeast-1.anthropic.model", "bedrock", "anthropic.model"),
            ("bedrock/converse/ap-northeast-1.anthropic.priority", "bedrock", "ap-northeast-1.anthropic.priority"),
            ("bedrock/converse/us.anthropic.model", "bedrock", "us.anthropic.model"),
            ("bedrock/converse/au.anthropic.dated-2026-01-01", "bedrock", "anthropic.dated-2026-01-01"),
            ("hosted_vllm/hosted_vllm/hosted_vllm/hosted", "hosted_vllm", "hosted"),
            ("openai/azure_ai/openai/model", "openai", "openai/model"),
            ("openai/openai/bedrock/us.anthropic.model", "bedrock", "us.anthropic.model"),
            ("openai/openai/bedrock/us.anthropic.model", None, None),
            ("bedrock/bedrock/bedrock/us.anthropic.model", "azure_ai", None),
            ("bedrock/unknown/bedrock/us.anthropic.model", "bedrock", None),
            ("bedrock/Bedrock/bedrock/us.anthropic.model", "bedrock", None),
            ("bedrock/AP-NORTHEAST-1.anthropic.model", "bedrock", None),
            ("bedrock/ap-northeast-1.anthropic.MODEL", "bedrock", None),
            ("bedrock/ap-northeast-1.anthropic.Model", "bedrock", "anthropic.Model"),
            ("bedrock/bedrock/bedrock/unmatched", "bedrock", None),
            ("bedrock/ap-northeast-2.anthropic.model", "bedrock", None),
            ("bedrock/bedrock/bedrock/us.anthropic.dated-2026-01-01", "bedrock", "anthropic.dated-2026-01-01"),
            ("azure_ai/model", None, "model"),
            ("bedrock/ap-northeast-1.anthropic.null", "bedrock", None),
            ("bedrock/bedrock/bedrock/anthropic.absent", "bedrock", None),
        )
        before: Final = json.dumps(rows)
        with patch.object(litellm, "model_cost", rows):
            for key, provider, expected in cases:
                with self.subTest(key=key, provider=provider):
                    self._assert_legacy_candidate(rows, key, provider, expected)
        self.assertEqual(json.dumps(rows), before)

    def _assert_legacy_candidate(
        self, rows: dict[str, object], key: str, provider: str | None, expected: str | None
    ) -> None:
        legacy: Final = _resolve_builtin_model_cost_entry(key, provider or "")
        self.assertEqual(None if legacy is None else legacy.key, expected)
        target: Final = native.registration_target(rows, key, provider=provider)
        self.assertEqual((target.write, target.inherit_from), (key, expected))
        self.assertIs(target.names_the_row, False if legacy is None else legacy.names_the_row)

    def test_repeated_prefix_names_row_and_preserves_fallback_price_variants(self) -> None:
        self._assert_fallback_price_variants("us.anthropic.model", "bedrock/bedrock/bedrock/us.anthropic.model", True)

    def test_region_only_alias_does_not_name_row_and_suppresses_fallback_price_variants(self) -> None:
        self._assert_fallback_price_variants("anthropic.model", "bedrock/ap-northeast-1.anthropic.model", False)

    def test_region_candidate_also_in_suffix_names_preserves_fallback_price_variants(self) -> None:
        self._assert_fallback_price_variants(
            "anthropic.model-2026-01-01", "bedrock/converse/us.anthropic.model-2026-01-01", True
        )

    def _assert_fallback_price_variants(self, source: str, key: str, names_the_row: bool) -> None:
        row: Final = {
            "litellm_provider": "bedrock_converse",
            "input_cost_per_token": 0.001,
            "output_cost_per_token": 0.002,
            "input_cost_per_token_above_128k_tokens": 0.01,
            "tiered_pricing": [
                {"input_cost_per_token": 0.003, "output_cost_per_token": 0.004, "range": [0, 256_000]},
                {"input_cost_per_token": 0.005, "output_cost_per_token": 0.006, "range": [256_000, 1_000_000]},
            ],
        }
        rows: Final[dict[str, object]] = {source: row}
        override: Final = {"input_cost_per_token": 0.007}
        before: Final = json.dumps(rows)
        with patch.object(litellm, "model_cost", rows):
            legacy: Final = _resolve_builtin_model_cost_entry(key, "bedrock")
            self.assertIsNotNone(legacy)
            if legacy is None:
                self.fail("synthetic source must match the M2 reference")
            self.assertEqual((legacy.key, legacy.names_the_row), (source, names_the_row))
            fallback: Final = _fallback_registration(key, "bedrock", override)
            self.assertTrue(fallback.matched_builtin)
            self.assertEqual("input_cost_per_token_above_128k_tokens" in fallback.entry, names_the_row)
            target: Final = native.registration_target(rows, key, provider="bedrock")
            self.assertEqual((target.write, target.inherit_from, target.names_the_row), (key, source, names_the_row))
            inherited: Final = row if target.names_the_row else _without_overridden_price_variants(row, override)
            self.assertEqual(
                fallback.entry, {name: value for name, value in inherited.items() if name != "litellm_provider"}
            )
        self.assertEqual(json.dumps(rows), before)

    def test_row_name_metadata_is_read_only_and_false_without_inheritance(self) -> None:
        rows: Final[dict[str, object]] = {"same": {"litellm_provider": "openai"}}
        target: Final = native.registration_target(rows, "same")
        self.assertIs(target.names_the_row, True)
        self.assertRaises(AttributeError, setattr, target, "names_the_row", False)
        self.assertIs(target.names_the_row, True)
        self.assertIs(native.registration_target(rows, "absent").names_the_row, False)
        self.assertIs(native.registration_target(rows, "same", provider="azure").names_the_row, False)

    def test_repeated_prefix_inherits_prices_without_expanding_read_resolution(self) -> None:
        self._assert_inherits_live_prices("us.anthropic.model", "bedrock/bedrock/bedrock/us.anthropic.model")

    def test_region_alias_inherits_prices_without_expanding_read_resolution(self) -> None:
        self._assert_inherits_live_prices("anthropic.flagless-model", "bedrock/ap-northeast-1.anthropic.flagless-model")

    def _assert_inherits_live_prices(self, source: str, key: str) -> None:
        row: Final = {
            "litellm_provider": "bedrock_converse",
            "input_cost_per_token": 0.001,
            "output_cost_per_token": 0.002,
            "cache_read_input_token_cost": 0.0001,
        }
        rows: Final[dict[str, object]] = {source: row}
        target: Final = native.registration_target(rows, key, provider="bedrock")
        self.assertEqual((target.write, target.inherit_from), (key, source))
        self.assertIsInstance(native.resolve_model(rows, key, provider="bedrock"), native.ModelNotFound)
        layers: Final = native.ModelCostLayers()
        fields: Final[dict[str, JsonValue]] = {"litellm_provider": "bedrock", "mode": "chat"}
        layers.record_registration(target.write, target.inherit_from, fields)
        base: Final = native.ModelRegistry.parse(json.dumps(rows).encode())
        self.assertEqual(
            layers.compose_keys(base, frozenset({key})),
            [(key, {**row, **fields}, frozenset(fields))],
        )
        self.assertEqual(layers.registrations(), [(key, source, fields)])
        row["input_cost_per_token"] = 0.004
        updated: Final = native.ModelRegistry.parse(json.dumps(rows).encode())
        self.assertEqual(
            layers.compose_keys(updated, frozenset({key})),
            [(key, {**row, **fields}, frozenset(fields))],
        )

    def test_provider_is_optional_and_keyword_only(self) -> None:
        rows: Final[dict[str, object]] = {"same": {"litellm_provider": "openai"}}
        old: Final = native.registration_target(rows, "same")
        explicit: Final = native.registration_target(rows, "same", provider=None)
        self.assertEqual((old.write, old.inherit_from), ("same", "same"))
        self.assertEqual((explicit.write, explicit.inherit_from), (old.write, old.inherit_from))
        self.assertIs(old.names_the_row, True)
        self.assertIs(explicit.names_the_row, True)
        provider: Final = inspect.signature(native.registration_target).parameters["provider"]
        self.assertEqual(provider.kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertIsNone(provider.default)
        self.assertRaises(TypeError, native.registration_target, rows, "same", "openai")

    def test_registration_rejects_wrong_provider_without_changing_the_write_key(self) -> None:
        row: Final = {"litellm_provider": "openai"}
        rows: Final[dict[str, object]] = {"same": row}
        target: Final = native.registration_target(rows, "same", provider="azure")
        self.assertEqual((target.write, target.inherit_from), ("same", None))
        row["litellm_provider"] = "azure"
        updated: Final = native.registration_target(rows, "same", provider="azure")
        self.assertEqual((updated.write, updated.inherit_from), ("same", "same"))

    def test_registration_skips_folded_candidate_and_prefers_exact_qualified_candidate(self) -> None:
        rows: Final[dict[str, object]] = {
            "openai/Model": {"litellm_provider": "openai"},
            "model": {"litellm_provider": "openai"},
        }
        target: Final = native.registration_target(rows, "model", provider="openai")
        self.assertEqual((target.write, target.inherit_from), ("model", "model"))
        rows["openai/model"] = {"litellm_provider": "openai"}
        qualified: Final = native.registration_target(rows, "model", provider="openai")
        self.assertEqual((qualified.write, qualified.inherit_from), ("model", "openai/model"))

    def test_case_variant_registration_does_not_inherit_folded_fields(self) -> None:
        rows: Final[dict[str, object]] = {
            "hosted_vllm/Example-Model": {"litellm_provider": "hosted_vllm", "supports_vision": True}
        }
        target: Final = native.registration_target(rows, "hosted_vllm/example-model", provider="hosted_vllm")
        self.assertEqual((target.write, target.inherit_from), ("hosted_vllm/example-model", None))
        resolved: Final = native.resolve_model(rows, target.write, provider="hosted_vllm")
        assert isinstance(resolved, native.ResolvedModel)
        self.assertEqual(resolved.via, "Lowercase")
        base: Final = native.ModelRegistry.parse(
            b'{"hosted_vllm/Example-Model":{"litellm_provider":"hosted_vllm","supports_vision":true}}'
        )
        layers: Final = native.ModelCostLayers()
        layers.record_registration(target.write, target.inherit_from, {"max_tokens": 32000})
        self.assertEqual(
            layers.compose(base),
            [
                (
                    "hosted_vllm/Example-Model",
                    {"litellm_provider": "hosted_vllm", "supports_vision": True},
                    frozenset(),
                ),
                ("hosted_vllm/example-model", {"max_tokens": 32000}, frozenset({"max_tokens"})),
            ],
        )


class RegistrationSnapshotTests(unittest.TestCase):
    def test_snapshots_keep_original_sparse_records_in_order(self) -> None:
        layers: Final = native.ModelCostLayers()
        self.assertEqual(layers.registrations(), [])
        layers.record_registration("z", "base", {"removed": None, "input_cost_per_token": 0})
        layers.diff({"z": ({"input_cost_per_token": 0}, {"input_cost_per_token": 9, "direct": True})})
        layers.record_registration("a", None, {})
        layers.record_registration("z", "a", {"supports_vision": False})
        layers.diff({"z": ({"direct": True}, None), "direct-only": (None, {"extra": 1})})
        expected: Final = [
            ("z", "base", {"removed": None, "input_cost_per_token": 0}),
            ("a", None, {}),
            ("z", "a", {"supports_vision": False}),
        ]
        snapshot: Final = layers.registrations()
        self.assertEqual(snapshot, expected)
        self.assertEqual(list(snapshot[0][2]), ["removed", "input_cost_per_token"])
        base: Final = native.ModelRegistry.parse(b'{"base":{"litellm_provider":"openai","max_tokens":42}}')
        self.assertEqual(layers.compose_keys(base, frozenset({"z"})), [("z", None, frozenset())])
        self.assertEqual(layers.registrations(), expected)
        layers.record_registration("later", "z", {"max_tokens": 7})
        self.assertEqual(snapshot, expected)
        self.assertEqual(layers.registrations(), [*expected, ("later", "z", {"max_tokens": 7})])
        layers.forget_writes()
        self.assertEqual(layers.registrations(), [])
        self.assertEqual(snapshot, expected)

    def test_snapshot_nested_values_are_detached_from_inputs_and_later_snapshots(self) -> None:
        layers: Final = native.ModelCostLayers()
        nested: Final = {"items": [{"value": 1, "null": None}]}
        layers.record_registration("model", None, {"metadata": nested, "remove": None})
        layers.record_registration("other", "model", {"metadata": nested})
        expected: Final = [
            ("model", None, {"metadata": {"items": [{"value": 1, "null": None}]}, "remove": None}),
            ("other", "model", {"metadata": {"items": [{"value": 1, "null": None}]}}),
        ]
        nested["items"][0]["value"] = 2
        first: Final = layers.registrations()
        second: Final = layers.registrations()
        self.assertEqual(first, expected)
        first[0][2]["metadata"]["items"][0]["value"] = 3
        first[0][2]["remove"] = "changed"
        first.pop()
        self.assertEqual(second, expected)
        self.assertEqual(layers.registrations(), expected)


if __name__ == "__main__":
    unittest.main()
