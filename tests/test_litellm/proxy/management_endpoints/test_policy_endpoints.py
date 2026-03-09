"""
Unit tests for policy management endpoints.

Tests apply_policies: resolving guardrails from policy names and applying them to inputs.
"""

from unittest.mock import MagicMock, patch

import pytest

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy.management_endpoints.policy_endpoints import apply_policies
from litellm.types.utils import GenericGuardrailAPIInputs


class _FakeGuardrailWithApply(CustomGuardrail):
    """Minimal CustomGuardrail subclass that defines apply_guardrail (in type.__dict__)."""

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: str,
        logging_obj=None,
    ) -> GenericGuardrailAPIInputs:
        return getattr(self, "_return_inputs", inputs)

    def set_return(self, return_inputs: GenericGuardrailAPIInputs) -> None:
        self._return_inputs = return_inputs


@pytest.fixture
def sample_inputs() -> GenericGuardrailAPIInputs:
    return {"texts": ["hello world"]}


@pytest.fixture
def request_data() -> dict:
    return {"model": "gpt-4"}


@pytest.fixture
def proxy_logging_obj():
    return MagicMock()


class TestApplyPoliciesEarlyReturn:
    """Test apply_policies when it returns inputs unchanged."""

    @pytest.mark.asyncio
    async def test_returns_inputs_unchanged_when_policy_names_none(
        self, sample_inputs, request_data, proxy_logging_obj
    ):
        result = await apply_policies(
            policy_names=None,
            inputs=sample_inputs,
            request_data=request_data,
            input_type="request",
            proxy_logging_obj=proxy_logging_obj,
        )
        assert result["inputs"] == sample_inputs
        assert result["guardrail_errors"] == []

    @pytest.mark.asyncio
    async def test_returns_inputs_unchanged_when_policy_names_empty(
        self, sample_inputs, request_data, proxy_logging_obj
    ):
        result = await apply_policies(
            policy_names=[],
            inputs=sample_inputs,
            request_data=request_data,
            input_type="request",
            proxy_logging_obj=proxy_logging_obj,
        )
        assert result["inputs"] == sample_inputs
        assert result["guardrail_errors"] == []

    @pytest.mark.asyncio
    async def test_returns_inputs_unchanged_when_both_policy_and_guardrail_names_empty(
        self, sample_inputs, request_data, proxy_logging_obj
    ):
        result = await apply_policies(
            policy_names=[],
            inputs=sample_inputs,
            request_data=request_data,
            input_type="request",
            proxy_logging_obj=proxy_logging_obj,
            guardrail_names=[],
        )
        assert result["inputs"] == sample_inputs
        assert result["guardrail_errors"] == []

    @pytest.mark.asyncio
    async def test_returns_inputs_unchanged_when_registry_not_initialized(
        self, sample_inputs, request_data, proxy_logging_obj
    ):
        mock_registry = MagicMock()
        mock_registry.is_initialized.return_value = False

        with patch(
            "litellm.proxy.management_endpoints.policy_endpoints.endpoints.get_policy_registry",
            return_value=mock_registry,
        ):
            result = await apply_policies(
                policy_names=["some-policy"],
                inputs=sample_inputs,
                request_data=request_data,
                input_type="request",
                proxy_logging_obj=proxy_logging_obj,
            )

        assert result["inputs"] == sample_inputs
        assert result["guardrail_errors"] == []
        mock_registry.is_initialized.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_inputs_unchanged_when_resolved_guardrails_empty(
        self, sample_inputs, request_data, proxy_logging_obj
    ):
        from litellm.types.proxy.policy_engine import ResolvedPolicy

        mock_registry = MagicMock()
        mock_registry.is_initialized.return_value = True
        mock_registry.get_all_policies.return_value = {}

        with patch(
            "litellm.proxy.management_endpoints.policy_endpoints.endpoints.get_policy_registry",
            return_value=mock_registry,
        ), patch(
            "litellm.proxy.management_endpoints.policy_endpoints.endpoints.PolicyResolver.resolve_policy_guardrails",
            return_value=ResolvedPolicy(policy_name="p", guardrails=[], inheritance_chain=[]),
        ):
            result = await apply_policies(
                policy_names=["empty-policy"],
                inputs=sample_inputs,
                request_data=request_data,
                input_type="request",
                proxy_logging_obj=proxy_logging_obj,
            )

        assert result["inputs"] == sample_inputs
        assert result["guardrail_errors"] == []


class TestApplyPoliciesWithGuardrails:
    """Test apply_policies when guardrails are resolved and applied."""

    @pytest.mark.asyncio
    async def test_applies_single_guardrail_and_returns_modified_inputs(
        self, sample_inputs, request_data, proxy_logging_obj
    ):
        from litellm.types.proxy.policy_engine import ResolvedPolicy

        mock_registry = MagicMock()
        mock_registry.is_initialized.return_value = True
        mock_registry.get_all_policies.return_value = {}

        modified_inputs: GenericGuardrailAPIInputs = {"texts": ["modified by guardrail"]}
        callback = _FakeGuardrailWithApply(guardrail_name="my_guardrail")
        callback.set_return(modified_inputs)

        mock_guardrail_registry = MagicMock()
        mock_guardrail_registry.get_initialized_guardrail_callback.return_value = callback

        with patch(
            "litellm.proxy.management_endpoints.policy_endpoints.endpoints.get_policy_registry",
            return_value=mock_registry,
        ), patch(
            "litellm.proxy.management_endpoints.policy_endpoints.endpoints.PolicyResolver.resolve_policy_guardrails",
            return_value=ResolvedPolicy(
                policy_name="p",
                guardrails=["my_guardrail"],
                inheritance_chain=["p"],
            ),
        ), patch(
            "litellm.proxy.management_endpoints.policy_endpoints.endpoints.GuardrailRegistry",
            return_value=mock_guardrail_registry,
        ):
            result = await apply_policies(
                policy_names=["my-policy"],
                inputs=sample_inputs,
                request_data=request_data,
                input_type="request",
                proxy_logging_obj=proxy_logging_obj,
            )

        assert result["inputs"] == modified_inputs
        assert result["guardrail_errors"] == []

    @pytest.mark.asyncio
    async def test_applies_multiple_guardrails_in_order(
        self, sample_inputs, request_data, proxy_logging_obj
    ):
        from litellm.types.proxy.policy_engine import ResolvedPolicy

        mock_registry = MagicMock()
        mock_registry.is_initialized.return_value = True
        mock_registry.get_all_policies.return_value = {}

        first_output: GenericGuardrailAPIInputs = {"texts": ["after first"]}
        second_output: GenericGuardrailAPIInputs = {"texts": ["after second"]}

        callback_a = _FakeGuardrailWithApply(guardrail_name="guardrail_a")
        callback_a.set_return(first_output)
        callback_b = _FakeGuardrailWithApply(guardrail_name="guardrail_b")
        callback_b.set_return(second_output)

        def get_callback(guardrail_name):
            if guardrail_name == "guardrail_a":
                return callback_a
            if guardrail_name == "guardrail_b":
                return callback_b
            return None

        mock_guardrail_registry = MagicMock()
        mock_guardrail_registry.get_initialized_guardrail_callback.side_effect = get_callback

        with patch(
            "litellm.proxy.management_endpoints.policy_endpoints.endpoints.get_policy_registry",
            return_value=mock_registry,
        ), patch(
            "litellm.proxy.management_endpoints.policy_endpoints.endpoints.PolicyResolver.resolve_policy_guardrails",
            return_value=ResolvedPolicy(
                policy_name="p",
                guardrails=["guardrail_a", "guardrail_b"],
                inheritance_chain=["p"],
            ),
        ), patch(
            "litellm.proxy.management_endpoints.policy_endpoints.endpoints.GuardrailRegistry",
            return_value=mock_guardrail_registry,
        ):
            result = await apply_policies(
                policy_names=["my-policy"],
                inputs=sample_inputs,
                request_data=request_data,
                input_type="response",
                proxy_logging_obj=proxy_logging_obj,
            )

        assert result["inputs"] == second_output
        assert result["guardrail_errors"] == []

    @pytest.mark.asyncio
    async def test_skips_missing_guardrail_callback(
        self, sample_inputs, request_data, proxy_logging_obj
    ):
        from litellm.types.proxy.policy_engine import ResolvedPolicy

        mock_registry = MagicMock()
        mock_registry.is_initialized.return_value = True
        mock_registry.get_all_policies.return_value = {}

        mock_guardrail_registry = MagicMock()
        mock_guardrail_registry.get_initialized_guardrail_callback.return_value = None

        with patch(
            "litellm.proxy.management_endpoints.policy_endpoints.endpoints.get_policy_registry",
            return_value=mock_registry,
        ), patch(
            "litellm.proxy.management_endpoints.policy_endpoints.endpoints.PolicyResolver.resolve_policy_guardrails",
            return_value=ResolvedPolicy(
                policy_name="p",
                guardrails=["missing_guardrail"],
                inheritance_chain=["p"],
            ),
        ), patch(
            "litellm.proxy.management_endpoints.policy_endpoints.endpoints.GuardrailRegistry",
            return_value=mock_guardrail_registry,
        ):
            result = await apply_policies(
                policy_names=["my-policy"],
                inputs=sample_inputs,
                request_data=request_data,
                input_type="request",
                proxy_logging_obj=proxy_logging_obj,
            )

        assert result["inputs"] == sample_inputs
        assert result["guardrail_errors"] == []

    @pytest.mark.asyncio
    async def test_records_guardrail_error_on_failure(
        self, sample_inputs, request_data, proxy_logging_obj
    ):
        """When a guardrail's apply_guardrail raises, error is recorded and inputs still returned."""
        from litellm.types.proxy.policy_engine import ResolvedPolicy

        mock_registry = MagicMock()
        mock_registry.is_initialized.return_value = True
        mock_registry.get_all_policies.return_value = {}

        callback = _FakeGuardrailWithApply(guardrail_name="failing_guardrail")

        async def _raise(inputs, request_data, input_type, logging_obj=None):
            raise ValueError("Content blocked: PII detected")

        callback.apply_guardrail = _raise

        mock_guardrail_registry = MagicMock()
        mock_guardrail_registry.get_initialized_guardrail_callback.return_value = (
            callback
        )

        with patch(
            "litellm.proxy.management_endpoints.policy_endpoints.endpoints.get_policy_registry",
            return_value=mock_registry,
        ), patch(
            "litellm.proxy.management_endpoints.policy_endpoints.endpoints.PolicyResolver.resolve_policy_guardrails",
            return_value=ResolvedPolicy(
                policy_name="p",
                guardrails=["failing_guardrail"],
                inheritance_chain=["p"],
            ),
        ), patch(
            "litellm.proxy.management_endpoints.policy_endpoints.endpoints.GuardrailRegistry",
            return_value=mock_guardrail_registry,
        ):
            result = await apply_policies(
                policy_names=["my-policy"],
                inputs=sample_inputs,
                request_data=request_data,
                input_type="request",
                proxy_logging_obj=proxy_logging_obj,
            )

        assert result["inputs"] == sample_inputs
        assert result["guardrail_errors"] == [
            {"guardrail_name": "failing_guardrail", "message": "Content blocked: PII detected"}
        ]

    @pytest.mark.asyncio
    async def test_skips_callback_without_apply_guardrail(
        self, sample_inputs, request_data, proxy_logging_obj
    ):
        """Guardrails that do not define apply_guardrail on their class are skipped."""
        from litellm.types.proxy.policy_engine import ResolvedPolicy

        class GuardrailWithoutApply(CustomGuardrail):
            """Subclass that does not override apply_guardrail (not in type(x).__dict__)."""
            pass

        callback_no_apply = GuardrailWithoutApply(guardrail_name="no_apply")
        assert "apply_guardrail" not in type(callback_no_apply).__dict__

        mock_registry = MagicMock()
        mock_registry.is_initialized.return_value = True
        mock_registry.get_all_policies.return_value = {}

        mock_guardrail_registry = MagicMock()
        mock_guardrail_registry.get_initialized_guardrail_callback.return_value = (
            callback_no_apply
        )

        with patch(
            "litellm.proxy.management_endpoints.policy_endpoints.endpoints.get_policy_registry",
            return_value=mock_registry,
        ), patch(
            "litellm.proxy.management_endpoints.policy_endpoints.endpoints.PolicyResolver.resolve_policy_guardrails",
            return_value=ResolvedPolicy(
                policy_name="p",
                guardrails=["no_apply_guardrail"],
                inheritance_chain=["p"],
            ),
        ), patch(
            "litellm.proxy.management_endpoints.policy_endpoints.endpoints.GuardrailRegistry",
            return_value=mock_guardrail_registry,
        ):
            result = await apply_policies(
                policy_names=["my-policy"],
                inputs=sample_inputs,
                request_data=request_data,
                input_type="request",
                proxy_logging_obj=proxy_logging_obj,
            )

        assert result["inputs"] == sample_inputs
        assert result["guardrail_errors"] == []

    @pytest.mark.asyncio
    async def test_collects_all_guardrail_failures_when_multiple_fail(
        self, sample_inputs, request_data, proxy_logging_obj
    ):
        """When multiple guardrails raise, all failures are collected and inputs still returned."""
        from litellm.types.proxy.policy_engine import ResolvedPolicy

        mock_registry = MagicMock()
        mock_registry.is_initialized.return_value = True
        mock_registry.get_all_policies.return_value = {}

        callback_a = _FakeGuardrailWithApply(guardrail_name="guardrail_a")

        async def _raise_a(inputs, request_data, input_type, logging_obj=None):
            raise ValueError("PII detected")

        callback_a.apply_guardrail = _raise_a

        callback_b = _FakeGuardrailWithApply(guardrail_name="guardrail_b")

        async def _raise_b(inputs, request_data, input_type, logging_obj=None):
            raise RuntimeError("Toxicity detected")

        callback_b.apply_guardrail = _raise_b

        def get_callback(guardrail_name):
            if guardrail_name == "guardrail_a":
                return callback_a
            if guardrail_name == "guardrail_b":
                return callback_b
            return None

        mock_guardrail_registry = MagicMock()
        mock_guardrail_registry.get_initialized_guardrail_callback.side_effect = (
            get_callback
        )

        with patch(
            "litellm.proxy.management_endpoints.policy_endpoints.endpoints.get_policy_registry",
            return_value=mock_registry,
        ), patch(
            "litellm.proxy.management_endpoints.policy_endpoints.endpoints.PolicyResolver.resolve_policy_guardrails",
            return_value=ResolvedPolicy(
                policy_name="p",
                guardrails=["guardrail_a", "guardrail_b"],
                inheritance_chain=["p"],
            ),
        ), patch(
            "litellm.proxy.management_endpoints.policy_endpoints.endpoints.GuardrailRegistry",
            return_value=mock_guardrail_registry,
        ):
            result = await apply_policies(
                policy_names=["my-policy"],
                inputs=sample_inputs,
                request_data=request_data,
                input_type="request",
                proxy_logging_obj=proxy_logging_obj,
            )

        assert result["inputs"] == sample_inputs
        assert len(result["guardrail_errors"]) == 2
        by_name = {e["guardrail_name"]: e["message"] for e in result["guardrail_errors"]}
        assert by_name["guardrail_a"] == "PII detected"
        assert by_name["guardrail_b"] == "Toxicity detected"


class TestApplyPoliciesMultiplePolicies:
    """Test apply_policies with multiple policy names (guardrail union)."""

    @pytest.mark.asyncio
    async def test_resolves_guardrails_from_multiple_policies(
        self, sample_inputs, request_data, proxy_logging_obj
    ):
        from litellm.types.proxy.policy_engine import ResolvedPolicy

        mock_registry = MagicMock()
        mock_registry.is_initialized.return_value = True
        mock_registry.get_all_policies.return_value = {}

        final_inputs: GenericGuardrailAPIInputs = {"texts": ["final"]}
        callback = _FakeGuardrailWithApply(guardrail_name="shared")
        callback.set_return(final_inputs)

        mock_guardrail_registry = MagicMock()
        mock_guardrail_registry.get_initialized_guardrail_callback.return_value = callback

        resolve_returns = [
            ResolvedPolicy(
                policy_name="policy_a",
                guardrails=["guardrail_1"],
                inheritance_chain=["policy_a"],
            ),
            ResolvedPolicy(
                policy_name="policy_b",
                guardrails=["guardrail_2"],
                inheritance_chain=["policy_b"],
            ),
        ]

        with patch(
            "litellm.proxy.management_endpoints.policy_endpoints.endpoints.get_policy_registry",
            return_value=mock_registry,
        ), patch(
            "litellm.proxy.management_endpoints.policy_endpoints.endpoints.PolicyResolver.resolve_policy_guardrails",
            side_effect=resolve_returns,
        ), patch(
            "litellm.proxy.management_endpoints.policy_endpoints.endpoints.GuardrailRegistry",
            return_value=mock_guardrail_registry,
        ):
            result = await apply_policies(
                policy_names=["policy_a", "policy_b"],
                inputs=sample_inputs,
                request_data=request_data,
                input_type="request",
                proxy_logging_obj=proxy_logging_obj,
            )

        assert result["inputs"] == final_inputs
        assert result["guardrail_errors"] == []


class TestApplyPoliciesDirectGuardrailNames:
    """Test apply_policies with direct guardrail_names (no policy registry)."""

    @pytest.mark.asyncio
    async def test_applies_guardrails_from_direct_guardrail_names_only(
        self, sample_inputs, request_data, proxy_logging_obj
    ):
        """When only guardrail_names is passed, policy registry is not used."""
        modified_inputs: GenericGuardrailAPIInputs = {"texts": ["from direct guardrail"]}
        callback = _FakeGuardrailWithApply(guardrail_name="my_guardrail")
        callback.set_return(modified_inputs)

        mock_guardrail_registry = MagicMock()
        mock_guardrail_registry.get_initialized_guardrail_callback.return_value = callback

        with patch(
            "litellm.proxy.management_endpoints.policy_endpoints.endpoints.GuardrailRegistry",
            return_value=mock_guardrail_registry,
        ):
            result = await apply_policies(
                policy_names=None,
                inputs=sample_inputs,
                request_data=request_data,
                input_type="request",
                proxy_logging_obj=proxy_logging_obj,
                guardrail_names=["my_guardrail"],
            )

        assert result["inputs"] == modified_inputs
        assert result["guardrail_errors"] == []
        mock_guardrail_registry.get_initialized_guardrail_callback.assert_called_once_with(
            guardrail_name="my_guardrail"
        )

    @pytest.mark.asyncio
    async def test_applies_guardrails_from_both_policy_names_and_guardrail_names(
        self, sample_inputs, request_data, proxy_logging_obj
    ):
        """Guardrails from policy_names and guardrail_names are merged and applied."""
        from litellm.types.proxy.policy_engine import ResolvedPolicy

        mock_registry = MagicMock()
        mock_registry.is_initialized.return_value = True
        mock_registry.get_all_policies.return_value = {}

        first_output: GenericGuardrailAPIInputs = {"texts": ["after first"]}
        second_output: GenericGuardrailAPIInputs = {"texts": ["after second"]}
        callback_from_policy = _FakeGuardrailWithApply(guardrail_name="from_policy")
        callback_from_policy.set_return(first_output)
        callback_direct = _FakeGuardrailWithApply(guardrail_name="direct_guardrail")
        callback_direct.set_return(second_output)

        def get_callback(guardrail_name):
            if guardrail_name == "from_policy":
                return callback_from_policy
            if guardrail_name == "direct_guardrail":
                return callback_direct
            return None

        mock_guardrail_registry = MagicMock()
        mock_guardrail_registry.get_initialized_guardrail_callback.side_effect = (
            get_callback
        )

        with patch(
            "litellm.proxy.management_endpoints.policy_endpoints.endpoints.get_policy_registry",
            return_value=mock_registry,
        ), patch(
            "litellm.proxy.management_endpoints.policy_endpoints.endpoints.PolicyResolver.resolve_policy_guardrails",
            return_value=ResolvedPolicy(
                policy_name="p",
                guardrails=["from_policy"],
                inheritance_chain=["p"],
            ),
        ), patch(
            "litellm.proxy.management_endpoints.policy_endpoints.endpoints.GuardrailRegistry",
            return_value=mock_guardrail_registry,
        ):
            result = await apply_policies(
                policy_names=["my-policy"],
                inputs=sample_inputs,
                request_data=request_data,
                input_type="request",
                proxy_logging_obj=proxy_logging_obj,
                guardrail_names=["direct_guardrail"],
            )

        # Sorted order: direct_guardrail then from_policy; final output is from_policy
        assert result["inputs"] == first_output
        assert result["guardrail_errors"] == []


# ---------------------------------------------------------------------------
# Tests for competitor enrichment helper functions
# ---------------------------------------------------------------------------
from litellm.proxy.management_endpoints.policy_endpoints import (
    _build_all_names_per_competitor,
    _build_comparison_blocked_words,
    _build_competitor_guardrail_definitions,
    _build_name_blocked_words,
    _build_recommendation_blocked_words,
    _build_refinement_prompt,
    _clean_competitor_line,
    _parse_variations_response,
)


class TestCleanCompetitorLine:
    """Tests for _clean_competitor_line."""

    def test_strips_bullets_and_dashes(self):
        assert _clean_competitor_line("- United Airlines") == "United Airlines"
        assert _clean_competitor_line("  - JetBlue  ") == "JetBlue"

    def test_strips_trailing_punctuation(self):
        assert _clean_competitor_line("Delta Airlines.") == "Delta Airlines"
        assert _clean_competitor_line("Southwest)") == "Southwest"

    def test_returns_none_for_empty(self):
        assert _clean_competitor_line("") is None
        assert _clean_competitor_line("   ") is None

    def test_returns_none_for_single_char(self):
        assert _clean_competitor_line("A") is None
        assert _clean_competitor_line(" - ") is None

    def test_plain_name(self):
        assert _clean_competitor_line("Qatar Airways") == "Qatar Airways"


class TestParseVariationsResponse:
    """Tests for _parse_variations_response."""

    def test_parses_standard_format(self):
        raw = "Delta Airlines: Delta Air Lines, DeltaAirlines, Delta\nUnited Airlines: United, UAL"
        competitors = ["Delta Airlines", "United Airlines"]
        result = _parse_variations_response(raw, competitors)
        assert "Delta Airlines" in result
        assert "Delta Air Lines" in result["Delta Airlines"]
        assert "United" in result["United Airlines"]

    def test_case_insensitive_matching(self):
        raw = "delta airlines: Delta Air Lines, DeltaAirlines"
        competitors = ["Delta Airlines"]
        result = _parse_variations_response(raw, competitors)
        assert "Delta Airlines" in result
        assert len(result["Delta Airlines"]) == 2

    def test_skips_lines_without_colon(self):
        raw = "This is a header\nDelta Airlines: Delta Air Lines"
        competitors = ["Delta Airlines"]
        result = _parse_variations_response(raw, competitors)
        assert len(result) == 1

    def test_skips_unknown_competitors(self):
        raw = "Unknown Corp: Foo, Bar\nDelta Airlines: Delta"
        competitors = ["Delta Airlines"]
        result = _parse_variations_response(raw, competitors)
        assert "Unknown Corp" not in result
        assert "Delta Airlines" in result

    def test_filters_out_self_reference(self):
        raw = "Delta Airlines: Delta Airlines, Delta Air Lines"
        competitors = ["Delta Airlines"]
        result = _parse_variations_response(raw, competitors)
        # "Delta Airlines" should be filtered out (same as canonical)
        assert "Delta Airlines" not in result["Delta Airlines"]
        assert "Delta Air Lines" in result["Delta Airlines"]

    def test_empty_input(self):
        assert _parse_variations_response("", []) == {}


class TestBuildRefinementPrompt:
    """Tests for _build_refinement_prompt."""

    def test_includes_brand_name(self):
        prompt = _build_refinement_prompt("add 10 more", ["Delta"], "Emirates")
        assert "Emirates" in prompt

    def test_includes_existing_competitors(self):
        prompt = _build_refinement_prompt("add more", ["Delta", "United"], "Emirates")
        assert "Delta" in prompt
        assert "United" in prompt

    def test_includes_instruction(self):
        prompt = _build_refinement_prompt("add 10 from Asia", ["Delta"], "Emirates")
        assert "add 10 from Asia" in prompt

    def test_asks_for_new_names_only(self):
        prompt = _build_refinement_prompt("add more", ["Delta"], "Emirates")
        assert "NEW" in prompt


class TestBuildAllNamesPerCompetitor:
    """Tests for _build_all_names_per_competitor."""

    def test_includes_canonical_and_variations(self):
        result = _build_all_names_per_competitor(
            ["Delta Airlines"], {"Delta Airlines": ["Delta", "DeltaAir"]}
        )
        assert result["Delta Airlines"] == ["Delta Airlines", "Delta", "DeltaAir"]

    def test_no_variations(self):
        result = _build_all_names_per_competitor(["Delta Airlines"], {})
        assert result["Delta Airlines"] == ["Delta Airlines"]

    def test_multiple_competitors(self):
        result = _build_all_names_per_competitor(
            ["Delta", "United"],
            {"Delta": ["DL"], "United": ["UA"]},
        )
        assert len(result) == 2
        assert result["Delta"] == ["Delta", "DL"]
        assert result["United"] == ["United", "UA"]


class TestBuildNameBlockedWords:
    """Tests for _build_name_blocked_words."""

    def test_basic_output(self):
        all_names = {"Delta": ["Delta", "DL"]}
        result = _build_name_blocked_words(["Delta"], all_names)
        keywords = [r["keyword"] for r in result]
        assert "Delta" in keywords
        assert "DL" in keywords
        assert all(r["action"] == "BLOCK" for r in result)

    def test_descriptions_differ_for_variations(self):
        all_names = {"Delta": ["Delta", "DL"]}
        result = _build_name_blocked_words(["Delta"], all_names)
        descs = {r["keyword"]: r["description"] for r in result}
        assert "Competitor: Delta" == descs["Delta"]
        assert "variation" in descs["DL"].lower()


class TestBuildRecommendationBlockedWords:
    """Tests for _build_recommendation_blocked_words."""

    def test_generates_prefix_combinations(self):
        all_names = {"Delta": ["Delta"]}
        result = _build_recommendation_blocked_words(["Delta"], all_names)
        keywords = [r["keyword"] for r in result]
        assert "try Delta" in keywords
        assert "use Delta" in keywords
        assert "switch to Delta" in keywords
        assert "consider Delta" in keywords

    def test_includes_variations(self):
        all_names = {"Delta": ["Delta", "DL"]}
        result = _build_recommendation_blocked_words(["Delta"], all_names)
        keywords = [r["keyword"] for r in result]
        assert "try DL" in keywords


class TestBuildComparisonBlockedWords:
    """Tests for _build_comparison_blocked_words."""

    def test_generates_competitor_comparisons(self):
        all_names = {"Delta": ["Delta"]}
        result = _build_comparison_blocked_words(["Delta"], all_names, "Emirates")
        keywords = [r["keyword"] for r in result]
        assert "Delta is better" in keywords

    def test_generates_brand_comparisons_once(self):
        all_names = {"Delta": ["Delta"], "United": ["United"]}
        result = _build_comparison_blocked_words(["Delta", "United"], all_names, "Emirates")
        keywords = [r["keyword"] for r in result]
        # Brand-level entries should appear exactly once
        assert keywords.count("better than Emirates") == 1
        assert keywords.count("Emirates is worse") == 1

    def test_includes_variation_comparisons(self):
        all_names = {"Delta": ["Delta", "DL"]}
        result = _build_comparison_blocked_words(["Delta"], all_names, "Emirates")
        keywords = [r["keyword"] for r in result]
        assert "DL is better" in keywords


class TestBuildCompetitorGuardrailDefinitions:
    """Tests for _build_competitor_guardrail_definitions."""

    def test_populates_blocked_words_for_known_guardrail_names(self):
        definitions = [
            {
                "guardrail_name": "competitor-name-blocker",
                "litellm_params": {"blocked_words": []},
            },
            {
                "guardrail_name": "competitor-recommendation-filter",
                "litellm_params": {"blocked_words": []},
            },
        ]
        result = _build_competitor_guardrail_definitions(
            definitions, ["Delta"], "Emirates", {"Delta": ["DL"]}
        )
        # Name blocker should have entries
        name_blocker = next(d for d in result if d["guardrail_name"] == "competitor-name-blocker")
        assert len(name_blocker["litellm_params"]["blocked_words"]) > 0

        # Recommendation filter should have entries
        rec_filter = next(d for d in result if d["guardrail_name"] == "competitor-recommendation-filter")
        assert len(rec_filter["litellm_params"]["blocked_words"]) > 0

    def test_does_not_modify_unknown_guardrail_names(self):
        definitions = [
            {
                "guardrail_name": "some-other-guardrail",
                "litellm_params": {"blocked_words": ["original"]},
            },
        ]
        result = _build_competitor_guardrail_definitions(
            definitions, ["Delta"], "Emirates"
        )
        assert result[0]["litellm_params"]["blocked_words"] == ["original"]

    def test_does_not_mutate_original_definitions(self):
        definitions = [
            {
                "guardrail_name": "competitor-name-blocker",
                "litellm_params": {"blocked_words": []},
            },
        ]
        _build_competitor_guardrail_definitions(definitions, ["Delta"], "Emirates")
        # Original should be unchanged
        assert definitions[0]["litellm_params"]["blocked_words"] == []

    def test_handles_input_and_output_blocker_variants(self):
        definitions = [
            {
                "guardrail_name": "competitor-name-input-blocker",
                "litellm_params": {"blocked_words": []},
            },
            {
                "guardrail_name": "competitor-name-output-blocker",
                "litellm_params": {"blocked_words": []},
            },
        ]
        result = _build_competitor_guardrail_definitions(
            definitions, ["Delta"], "Emirates"
        )
        for defn in result:
            assert len(defn["litellm_params"]["blocked_words"]) > 0
