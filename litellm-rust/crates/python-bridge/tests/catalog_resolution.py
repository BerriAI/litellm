from __future__ import annotations

import inspect
import unittest
from typing import Final
from unittest.mock import patch

import litellm
from litellm import utils
from litellm.rust_bridge import _native as native


class ModelInfoResolutionTests(unittest.TestCase):
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
            patch.object(litellm, "open_ai_chat_completion_models", [model]),
            patch.object(utils, "_model_cost_lowercase_map", None),
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
            patch.object(litellm, "cohere_chat_models", ["command-r-plus"]),
            patch.object(utils, "_model_cost_lowercase_map", None),
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
            patch.object(litellm, "ai21_chat_models", ["j2-light"]),
            patch.object(utils, "_model_cost_lowercase_map", None),
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
