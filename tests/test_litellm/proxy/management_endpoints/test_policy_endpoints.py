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
        assert result == sample_inputs

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
        assert result == sample_inputs

    @pytest.mark.asyncio
    async def test_returns_inputs_unchanged_when_registry_not_initialized(
        self, sample_inputs, request_data, proxy_logging_obj
    ):
        mock_registry = MagicMock()
        mock_registry.is_initialized.return_value = False

        with patch(
            "litellm.proxy.management_endpoints.policy_endpoints.get_policy_registry",
            return_value=mock_registry,
        ):
            result = await apply_policies(
                policy_names=["some-policy"],
                inputs=sample_inputs,
                request_data=request_data,
                input_type="request",
                proxy_logging_obj=proxy_logging_obj,
            )

        assert result == sample_inputs
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
            "litellm.proxy.management_endpoints.policy_endpoints.get_policy_registry",
            return_value=mock_registry,
        ), patch(
            "litellm.proxy.management_endpoints.policy_endpoints.PolicyResolver.resolve_policy_guardrails",
            return_value=ResolvedPolicy(policy_name="p", guardrails=[], inheritance_chain=[]),
        ):
            result = await apply_policies(
                policy_names=["empty-policy"],
                inputs=sample_inputs,
                request_data=request_data,
                input_type="request",
                proxy_logging_obj=proxy_logging_obj,
            )

        assert result == sample_inputs


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
            "litellm.proxy.management_endpoints.policy_endpoints.get_policy_registry",
            return_value=mock_registry,
        ), patch(
            "litellm.proxy.management_endpoints.policy_endpoints.PolicyResolver.resolve_policy_guardrails",
            return_value=ResolvedPolicy(
                policy_name="p",
                guardrails=["my_guardrail"],
                inheritance_chain=["p"],
            ),
        ), patch(
            "litellm.proxy.management_endpoints.policy_endpoints.GuardrailRegistry",
            return_value=mock_guardrail_registry,
        ):
            result = await apply_policies(
                policy_names=["my-policy"],
                inputs=sample_inputs,
                request_data=request_data,
                input_type="request",
                proxy_logging_obj=proxy_logging_obj,
            )

        assert result == modified_inputs

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
            "litellm.proxy.management_endpoints.policy_endpoints.get_policy_registry",
            return_value=mock_registry,
        ), patch(
            "litellm.proxy.management_endpoints.policy_endpoints.PolicyResolver.resolve_policy_guardrails",
            return_value=ResolvedPolicy(
                policy_name="p",
                guardrails=["guardrail_a", "guardrail_b"],
                inheritance_chain=["p"],
            ),
        ), patch(
            "litellm.proxy.management_endpoints.policy_endpoints.GuardrailRegistry",
            return_value=mock_guardrail_registry,
        ):
            result = await apply_policies(
                policy_names=["my-policy"],
                inputs=sample_inputs,
                request_data=request_data,
                input_type="response",
                proxy_logging_obj=proxy_logging_obj,
            )

        assert result == second_output

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
            "litellm.proxy.management_endpoints.policy_endpoints.get_policy_registry",
            return_value=mock_registry,
        ), patch(
            "litellm.proxy.management_endpoints.policy_endpoints.PolicyResolver.resolve_policy_guardrails",
            return_value=ResolvedPolicy(
                policy_name="p",
                guardrails=["missing_guardrail"],
                inheritance_chain=["p"],
            ),
        ), patch(
            "litellm.proxy.management_endpoints.policy_endpoints.GuardrailRegistry",
            return_value=mock_guardrail_registry,
        ):
            result = await apply_policies(
                policy_names=["my-policy"],
                inputs=sample_inputs,
                request_data=request_data,
                input_type="request",
                proxy_logging_obj=proxy_logging_obj,
            )

        assert result == sample_inputs

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
            "litellm.proxy.management_endpoints.policy_endpoints.get_policy_registry",
            return_value=mock_registry,
        ), patch(
            "litellm.proxy.management_endpoints.policy_endpoints.PolicyResolver.resolve_policy_guardrails",
            return_value=ResolvedPolicy(
                policy_name="p",
                guardrails=["no_apply_guardrail"],
                inheritance_chain=["p"],
            ),
        ), patch(
            "litellm.proxy.management_endpoints.policy_endpoints.GuardrailRegistry",
            return_value=mock_guardrail_registry,
        ):
            result = await apply_policies(
                policy_names=["my-policy"],
                inputs=sample_inputs,
                request_data=request_data,
                input_type="request",
                proxy_logging_obj=proxy_logging_obj,
            )

        assert result == sample_inputs


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
            "litellm.proxy.management_endpoints.policy_endpoints.get_policy_registry",
            return_value=mock_registry,
        ), patch(
            "litellm.proxy.management_endpoints.policy_endpoints.PolicyResolver.resolve_policy_guardrails",
            side_effect=resolve_returns,
        ), patch(
            "litellm.proxy.management_endpoints.policy_endpoints.GuardrailRegistry",
            return_value=mock_guardrail_registry,
        ):
            result = await apply_policies(
                policy_names=["policy_a", "policy_b"],
                inputs=sample_inputs,
                request_data=request_data,
                input_type="request",
                proxy_logging_obj=proxy_logging_obj,
            )

        assert result == final_inputs
