from __future__ import annotations

import inspect
import unittest
from collections import UserString
from typing import Final
from unittest.mock import patch

import litellm
from litellm import utils
from litellm.rust_bridge import _native as native


class ModelInfoResolutionTests(unittest.TestCase):
    def test_provider_callback_control_flow_is_not_a_lookup_failure(self) -> None:
        class InterruptedProvider:
            def __ne__(self, other: object) -> bool:
                raise KeyboardInterrupt

        rows: Final[dict[str, object]] = {"model": {"litellm_provider": InterruptedProvider()}}
        with self.assertRaises(KeyboardInterrupt):
            native.resolve_model(rows, "model", provider="openai", info_read=True)

    def test_info_read_preserves_m3_provider_protocol(self) -> None:
        values: Final[tuple[object, ...]] = (
            1,
            True,
            False,
            [],
            {},
            (),
            1.25,
            b"openai",
            bytearray(b"openai"),
            object(),
            UserString("openai"),
            UserString("bedrock_converse"),
            None,
            "",
            "openai",
            "bedrock_converse",
        )
        for provider in (
            "openai",
            "azure_ai",
            "bedrock",
            "bedrock_converse",
            "vertex_ai",
            "fireworks_ai",
            "github",
            "litellm_proxy",
            "nadir",
        ):
            for value in values:
                for explicit in (None, "", provider):
                    with self.subTest(provider=provider, value=value, explicit=explicit):
                        self.assert_provider_protocol(provider, value, explicit)

    def assert_provider_protocol(self, provider: str, value: object, explicit: str | None) -> None:
        model: Final = f"{provider}/invalid-model"
        first: Final = {"litellm_provider": value, "mode": "chat", "supports_vision": False}
        rows: Final[dict[str, object]] = {
            model: first,
            "invalid-model": {"litellm_provider": provider, "mode": "chat", "supports_vision": True},
        }
        result: Final = native.resolve_model(
            rows, model, provider=explicit, inferred=("invalid-model", provider), info_read=True
        )
        try:
            matches: Final = utils._check_provider_match(first, provider)  # pyright: ignore[reportUnknownMemberType, reportPrivateUsage]  # Exact M3 oracle has an untyped private boundary
        except Exception:
            self.assertIsInstance(result, native.InvalidProvider)
            if not isinstance(result, native.InvalidProvider):
                self.fail("provider comparison error became a lookup result")
            self.assertEqual((result.key, result.provider), (model, provider))
        else:
            self.assertIsInstance(result, native.ResolvedModel)
            if not isinstance(result, native.ResolvedModel):
                self.fail("matching candidate was lost")
            self.assertEqual(result.key, model if matches else "invalid-model")
        self.assertIs(rows[model], first)
        self.assertIs(first["litellm_provider"], value)

    def test_info_read_matches_public_m3_invalid_provider_results(self) -> None:
        for provider in ("openai", "bedrock", "vertex_ai", "fireworks_ai", "github"):
            for explicit in (None, "", provider):
                with self.subTest(provider=provider, explicit=explicit):
                    self.assert_public_m3_invalid_provider(provider, explicit)

    def assert_public_m3_invalid_provider(self, provider: str, explicit: str | None) -> None:
        model: Final = f"{provider}/invalid-model"
        rows: Final[dict[str, object]] = {
            model: {"litellm_provider": 1, "mode": "chat"},
            "invalid-model": {"litellm_provider": provider, "mode": "chat"},
        }
        with patch.object(litellm, "model_cost", rows):
            result: Final = native.resolve_model(
                rows, model, provider=explicit, inferred=("invalid-model", provider), info_read=True
            )
            if provider in ("bedrock", "vertex_ai", "fireworks_ai"):
                with self.assertRaisesRegex(Exception, "isn't mapped yet"):
                    utils.get_model_info_helper(model, custom_llm_provider=explicit)
                self.assertIsInstance(result, native.InvalidProvider)
            else:
                legacy: Final = utils.get_model_info_helper(model, custom_llm_provider=explicit)
                if not isinstance(result, native.ResolvedModel):
                    self.fail("public M3 selected a row")
                self.assertEqual(result.key, legacy["key"])

    def test_info_read_sees_mutations_without_changing_registration(self) -> None:
        row: Final[dict[str, object]] = {"litellm_provider": 1, "opaque": object()}
        rows: Final[dict[str, object]] = {"bedrock/model": row, "model": {"litellm_provider": "bedrock"}}
        before: Final = native.registration_target(rows, "bedrock/model", provider="bedrock")
        self.assertIsInstance(native.resolve_model(rows, "bedrock/model", provider="bedrock"), native.ResolvedModel)
        self.assertIsInstance(
            native.resolve_model(rows, "bedrock/model", provider="bedrock", info_read=True), native.InvalidProvider
        )
        row["litellm_provider"] = "openai"
        selected: Final = native.resolve_model(rows, "bedrock/model", provider="bedrock", info_read=True)
        if not isinstance(selected, native.ResolvedModel):
            self.fail("later valid row lost after provider mutation")
        self.assertEqual(selected.key, "model")
        row["litellm_provider"] = 1
        after: Final = native.registration_target(rows, "bedrock/model", provider="bedrock")
        self.assertEqual(
            (before.write, before.inherit_from, before.names_the_row),
            (after.write, after.inherit_from, after.names_the_row),
        )
        self.assertIsInstance(native.model_info(rows, "bedrock/model"), native.RowErrors)
        values: Final[tuple[object, ...]] = (None, 1, True, [], {}, object())
        for value in values:
            row["litellm_provider"] = value
            self.assertIsInstance(
                native.resolve_model(rows, "bedrock/model", inferred=("bedrock/model", None), info_read=True),
                native.ResolvedModel,
            )
        del row["litellm_provider"]
        self.assertIsInstance(
            native.resolve_model(rows, "bedrock/model", provider="bedrock", info_read=True), native.ResolvedModel
        )

    def test_sdk_explicit_bedrock_suffix_and_priority_match_m3(self) -> None:
        for model in ("probe-model-20260101", "probe-model"):
            with self.subTest(model=model):
                self.assert_sdk_bedrock_selection(model)

    def assert_sdk_bedrock_selection(self, model: str) -> None:
        rows: Final[dict[str, object]] = {
            "probe-model": {"litellm_provider": "bedrock", "mode": "chat", "supports_function_calling": True},
            "bedrock/probe-model": {"litellm_provider": "bedrock", "mode": "chat", "supports_function_calling": False},
        }
        with patch.object(litellm, "model_cost", rows):
            result: Final = native.resolve_model(
                rows, model, provider="bedrock", inferred=("ignored", "openai"), info_read=True
            )
            self.assertIsInstance(native.resolve_model(rows, model, provider="bedrock"), native.ResolvedModel)
            if model.endswith("-20260101"):
                with self.assertRaisesRegex(Exception, "isn't mapped yet"):
                    utils.get_model_info_helper(model, custom_llm_provider="bedrock")
                self.assertIsInstance(result, native.ModelNotFound)
                if not isinstance(result, native.ModelNotFound):
                    self.fail("SDK dated miss became a match")
                self.assertEqual(result.tried, [f"bedrock/{model}", model])
            else:
                legacy: Final = utils.get_model_info_helper(model, custom_llm_provider="bedrock")
                if not isinstance(result, native.ResolvedModel):
                    self.fail("qualified candidate was lost")
                self.assertEqual((result.key, legacy["key"]), ("bedrock/probe-model", "bedrock/probe-model"))

    def test_inferred_reads_match_m3_selection(self) -> None:
        cases: Final = (
            ("read-model", "read-model", "openai", "read-model", "openai/read-model"),
            ("read-model-2026-01-01", "read-model-2026-01-01", "openai", "read-model", "openai/read-model"),
            ("azure/read-model", "read-model", "azure", "read-model", "azure/read-model"),
            ("azure/READ-MODEL", "READ-MODEL", "azure", "read-model", "azure/READ-MODEL"),
            ("openai/read-model-2026-01-01", "read-model-2026-01-01", "openai", "openai/read-model", "read-model"),
            (
                "bedrock/converse/anthropic.read-model",
                "converse/anthropic.read-model",
                "bedrock",
                "anthropic.read-model",
                "bedrock/anthropic.read-model",
            ),
            (
                "bedrock/us.anthropic.read-model",
                "us.anthropic.read-model",
                "bedrock",
                "anthropic.read-model",
                "bedrock/anthropic.read-model",
            ),
            (
                "bedrock_mantle/us-east-1/read-model",
                "read-model",
                "bedrock_mantle",
                "bedrock_mantle/read-model",
                "read-model",
            ),
            ("vertex_ai/read-model-001", "read-model-001", "vertex_ai", "vertex_ai/read-model", "read-model"),
        )
        for model, normalized, provider, expected, other in cases:
            with self.subTest(model=model):
                self.assert_m3_selection(model, normalized, provider, expected, other)

    def assert_m3_selection(self, model: str, normalized: str, provider: str, expected: str, other: str) -> None:
        rows: Final[dict[str, object]] = {
            expected: {"litellm_provider": provider, "mode": "chat", "supports_vision": True},
            other: {
                "litellm_provider": "openai" if provider == "azure" else provider,
                "mode": "chat",
                "supports_vision": False,
            },
        }
        with (
            patch.object(litellm, "model_cost", rows),
            patch.object(litellm, "open_ai_chat_completion_models", {model}),
        ):
            native.invalidate_model_cost_index()
            routed_model, routed_provider, _, _ = litellm.get_llm_provider(model)
            self.assertEqual((routed_model, routed_provider), (normalized, provider))
            legacy: Final = utils.get_model_info_helper(model)
            selected: Final = native.resolve_model(rows, model, inferred=(routed_model, routed_provider))
            self.assertIsInstance(selected, native.ResolvedModel)
            if not isinstance(selected, native.ResolvedModel):
                self.fail(str(selected.tried))
            self.assertEqual((legacy["key"], selected.key), (expected, expected))
            info_selected: Final = native.resolve_model(
                rows, model, inferred=(routed_model, routed_provider), info_read=True
            )
            if not isinstance(info_selected, native.ResolvedModel):
                self.fail("info-read intent changed an inferred selection")
            self.assertEqual(info_selected.key, expected)
            lowered: Final = native.model_info(rows, selected.key)
            self.assertIsInstance(lowered, native.ModelInfoView)
            if not isinstance(lowered, native.ModelInfoView):
                self.fail(str(lowered.errors))
            self.assertEqual((legacy.get("supports_vision"), lowered.capabilities["supports_vision"]), (True, True))

    def test_host_normalization_can_preserve_azure_prefix(self) -> None:
        model: Final = "azure/command-r-plus"
        rows: Final[dict[str, object]] = {"command-r-plus": {"litellm_provider": "openai", "mode": "chat"}}
        with (
            patch.object(litellm, "model_cost", rows),
            patch.object(litellm, "cohere_chat_models", {"command-r-plus"}),
        ):
            normalized, provider, _, _ = litellm.get_llm_provider(model)
            self.assertEqual((normalized, provider), (model, "openai"))
            self.assertEqual(utils.get_potential_model_names(model, None)["split_model"], model)
            with self.assertRaisesRegex(Exception, "isn't mapped yet"):
                utils.get_model_info_helper(model)
            selected: Final = native.resolve_model(rows, model, inferred=(normalized, provider))
            self.assertIsInstance(selected, native.ModelNotFound)
            if not isinstance(selected, native.ModelNotFound):
                self.fail(selected.key)
            self.assertEqual(selected.tried, [model])

    def test_provider_mismatch_is_not_accepted(self) -> None:
        rows: Final[dict[str, object]] = {"j2-light": {"litellm_provider": "ai21", "mode": "chat"}}
        with (
            patch.object(litellm, "model_cost", rows),
            patch.object(litellm, "ai21_chat_models", {"j2-light"}),
        ):
            normalized, provider, _, _ = litellm.get_llm_provider("j2-light")
            self.assertEqual(provider, "ai21_chat")
            with self.assertRaisesRegex(Exception, "isn't mapped yet"):
                utils.get_model_info_helper("j2-light")
            self.assertIsInstance(
                native.resolve_model(rows, "j2-light", inferred=(normalized, provider)), native.ModelNotFound
            )
            self.assertIsInstance(native.resolve_model(rows, "j2-light"), native.ResolvedModel)

    def test_legacy_calls_and_registration_do_not_inherit_read_intent(self) -> None:
        rows: Final[dict[str, object]] = {
            "read-model": {"litellm_provider": "azure"},
            "azure/read-model": {"litellm_provider": "openai"},
            "openai/read-model": {"litellm_provider": "openai"},
        }
        cases: Final = (
            ("read-model", None, None, "read-model"),
            ("azure/read-model", None, None, "azure/read-model"),
            ("read-model", "openai", ("ignored", "azure"), "openai/read-model"),
            ("read-model", "", ("read-model", "azure"), "read-model"),
            ("azure/read-model", None, ("read-model", "azure"), "read-model"),
        )
        before: Final = native.registration_target(rows, "azure/read-model", provider="azure")
        for model, provider, inferred, expected in cases:
            with self.subTest(model=model, provider=provider, inferred=inferred):
                if not isinstance(
                    selected := native.resolve_model(rows, model, provider=provider, inferred=inferred),
                    native.ResolvedModel,
                ):
                    self.fail(str(selected.tried))
                self.assertEqual(selected.key, expected)
        after: Final = native.registration_target(rows, "azure/read-model", provider="azure")
        self.assertEqual(
            (before.write, before.inherit_from, before.names_the_row),
            (after.write, after.inherit_from, after.names_the_row),
        )
        self.assertEqual(
            (after.write, after.inherit_from, after.names_the_row), ("azure/read-model", "read-model", True)
        )

    def test_failed_inference_is_distinct_from_omitting_inference(self) -> None:
        rows: Final[dict[str, object]] = {"read-model": {}}
        self.assertIsInstance(native.resolve_model(rows, "unknown/read-model"), native.ResolvedModel)
        selected: Final = native.resolve_model(rows, "unknown/read-model", inferred=("unknown/read-model", None))
        self.assertIsInstance(selected, native.ModelNotFound)
        if not isinstance(selected, native.ModelNotFound):
            self.fail(selected.key)
        self.assertEqual(selected.tried, ["unknown/read-model"])

    def test_inferred_binding_shape(self) -> None:
        inferred: Final = inspect.signature(native.resolve_model).parameters["inferred"]
        self.assertEqual(inferred.kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertIn("inferred=None", str(inspect.signature(native.resolve_model)))
        for invalid in ("openai", ("model",), ("model", "openai", None), (None, "openai"), ("model", 123)):
            with self.subTest(invalid=invalid):
                self.assertRaises((TypeError, ValueError), native.resolve_model, {}, "model", inferred=invalid)


if __name__ == "__main__":
    unittest.main()
