"""
Tests for KeyManagementEventHooks.

Validates that email and secret manager operations are independent and non-blocking.
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.proxy.hooks.key_management_event_hooks import KeyManagementEventHooks

# Imported lazily inside the test classes below — pulling it in at module top
# would fail when the production helper is removed by a regression-bisect.
_obfuscate_audit_log_object_id = None


class TestKeyManagementEventHooksIndependentOperations:
    """Tests that email and secret manager operations are independent."""

    @pytest.mark.asyncio
    async def test_email_failure_does_not_block_secret_manager(self):
        """
        Test that if email sending fails, secret manager operation still runs.

        This validates the independent operation design where one failure
        does not block the other operation.
        """
        secret_manager_called = {"called": False}

        # Mock the email method to raise an exception
        async def mock_send_email_raises(*args, **kwargs):
            raise Exception("Email service unavailable")

        # Mock the secret manager method to track if it was called
        async def mock_store_secret(*args, **kwargs):
            secret_manager_called["called"] = True

        # Create mock objects for the hook parameters
        mock_data = MagicMock()
        mock_data.key_alias = "test-key-alias"
        mock_data.team_id = None
        mock_data.send_invite_email = True

        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "key": "sk-test",
            "token": "test-token",
        }
        mock_response.model_dump_json.return_value = '{"key": "sk-test"}'
        mock_response.token_id = "token-123"
        mock_response.key = "sk-test-key"

        mock_user_api_key_dict = MagicMock()
        mock_user_api_key_dict.user_id = "user-123"
        mock_user_api_key_dict.api_key = "api-key-123"

        with (
            patch.object(
                KeyManagementEventHooks,
                "_send_key_created_email",
                side_effect=mock_send_email_raises,
            ),
            patch.object(
                KeyManagementEventHooks,
                "_store_virtual_key_in_secret_manager",
                side_effect=mock_store_secret,
            ),
            patch.object(
                KeyManagementEventHooks,
                "_is_email_sending_enabled",
                return_value=True,
            ),
            patch("litellm.store_audit_logs", False),
            patch("litellm.proxy.hooks.key_management_event_hooks.verbose_proxy_logger"),
        ):
            # Should not raise even though email fails
            await KeyManagementEventHooks.async_key_generated_hook(
                data=mock_data,
                response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )

        # Secret manager should have been called despite email failure
        assert secret_manager_called["called"] is True

    @pytest.mark.asyncio
    async def test_secret_manager_failure_does_not_block_email(self):
        """
        Test that if secret manager fails, email operation still runs.

        This validates the independent operation design where one failure
        does not block the other operation.
        """
        email_called = {"called": False}

        # Mock the email method to track if it was called
        async def mock_send_email(*args, **kwargs):
            email_called["called"] = True

        # Mock the secret manager method to raise an exception
        async def mock_store_secret_raises(*args, **kwargs):
            raise Exception("Secret manager unavailable")

        # Create mock objects for the hook parameters
        mock_data = MagicMock()
        mock_data.key_alias = "test-key-alias"
        mock_data.team_id = None
        mock_data.send_invite_email = True

        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "key": "sk-test",
            "token": "test-token",
        }
        mock_response.model_dump_json.return_value = '{"key": "sk-test"}'
        mock_response.token_id = "token-123"
        mock_response.key = "sk-test-key"

        mock_user_api_key_dict = MagicMock()
        mock_user_api_key_dict.user_id = "user-123"
        mock_user_api_key_dict.api_key = "api-key-123"

        with (
            patch.object(
                KeyManagementEventHooks,
                "_send_key_created_email",
                side_effect=mock_send_email,
            ),
            patch.object(
                KeyManagementEventHooks,
                "_store_virtual_key_in_secret_manager",
                side_effect=mock_store_secret_raises,
            ),
            patch.object(
                KeyManagementEventHooks,
                "_is_email_sending_enabled",
                return_value=True,
            ),
            patch("litellm.store_audit_logs", False),
            patch("litellm.proxy.hooks.key_management_event_hooks.verbose_proxy_logger"),
        ):
            # Should not raise even though secret manager fails
            await KeyManagementEventHooks.async_key_generated_hook(
                data=mock_data,
                response=mock_response,
                user_api_key_dict=mock_user_api_key_dict,
            )

        # Email should have been called despite secret manager failure
        assert email_called["called"] is True


class TestRotateVirtualKeyInSecretManager:
    """Tests for _rotate_virtual_key_in_secret_manager with team_id support."""

    @pytest.mark.asyncio
    async def test_rotate_virtual_key_with_team_id(self):
        """Test that team_id is passed to async_rotate_secret."""
        from litellm.types.secret_managers.main import (
            KeyManagementSystem,
            KeyManagementSettings,
        )
        from litellm.secret_managers.base_secret_manager import BaseSecretManager
        import litellm

        # Setup - Create a mock that inherits from BaseSecretManager
        mock_secret_manager = MagicMock(spec=BaseSecretManager)
        mock_secret_manager.async_rotate_secret = AsyncMock(return_value={"status": "success"})

        litellm.secret_manager_client = mock_secret_manager
        litellm._key_management_system = KeyManagementSystem.HASHICORP_VAULT
        litellm._key_management_settings = KeyManagementSettings(
            store_virtual_keys=True,
            prefix_for_stored_virtual_keys="litellm/",
        )

        current_secret_name = "virtual-key-old"
        new_secret_name = "virtual-key-new"
        new_secret_value = "sk-new-key-value"
        team_id = "team-123"

        # Mock _get_secret_manager_optional_params to return team settings
        team_settings = {
            "namespace": "team-namespace",
            "mount": "kv-team",
            "path_prefix": "teams/custom",
        }

        # Patch isinstance in the key_management_event_hooks module to return True for BaseSecretManager check
        import builtins

        original_isinstance = builtins.isinstance

        def mock_isinstance(obj, cls):
            if cls == BaseSecretManager and obj == mock_secret_manager:
                return True
            return original_isinstance(obj, cls)

        with (
            patch.object(
                KeyManagementEventHooks,
                "_get_secret_manager_optional_params",
                return_value=team_settings,
            ) as mock_get_params,
            patch(
                "litellm.proxy.hooks.key_management_event_hooks.isinstance",
                side_effect=mock_isinstance,
            ),
        ):
            await KeyManagementEventHooks._rotate_virtual_key_in_secret_manager(
                current_secret_name=current_secret_name,
                new_secret_name=new_secret_name,
                new_secret_value=new_secret_value,
                team_id=team_id,
            )

            # Verify _get_secret_manager_optional_params was called with team_id
            mock_get_params.assert_called_once_with(team_id)

            # Verify async_rotate_secret was called with correct parameters
            mock_secret_manager.async_rotate_secret.assert_called_once()
            call_kwargs = mock_secret_manager.async_rotate_secret.call_args[1]

            # Verify secret names have prefix
            assert call_kwargs["current_secret_name"] == "litellm/virtual-key-old"
            assert call_kwargs["new_secret_name"] == "litellm/virtual-key-new"
            assert call_kwargs["new_secret_value"] == new_secret_value
            assert call_kwargs["optional_params"] == team_settings

    @pytest.mark.asyncio
    async def test_rotate_virtual_key_without_team_id(self):
        """Test that None team_id is handled correctly."""
        from litellm.types.secret_managers.main import (
            KeyManagementSystem,
            KeyManagementSettings,
        )
        from litellm.secret_managers.base_secret_manager import BaseSecretManager
        import litellm

        # Setup - Create a mock that inherits from BaseSecretManager
        mock_secret_manager = MagicMock(spec=BaseSecretManager)
        mock_secret_manager.async_rotate_secret = AsyncMock(return_value={"status": "success"})

        litellm.secret_manager_client = mock_secret_manager
        litellm._key_management_system = KeyManagementSystem.HASHICORP_VAULT
        litellm._key_management_settings = KeyManagementSettings(
            store_virtual_keys=True,
            prefix_for_stored_virtual_keys="litellm/",
        )

        current_secret_name = "virtual-key-old"
        new_secret_name = "virtual-key-new"
        new_secret_value = "sk-new-key-value"

        # Patch isinstance in the key_management_event_hooks module to return True for BaseSecretManager check
        import builtins

        original_isinstance = builtins.isinstance

        def mock_isinstance(obj, cls):
            if cls == BaseSecretManager and obj == mock_secret_manager:
                return True
            return original_isinstance(obj, cls)

        # Mock _get_secret_manager_optional_params to return None (no team settings)
        with (
            patch.object(
                KeyManagementEventHooks,
                "_get_secret_manager_optional_params",
                return_value=None,
            ) as mock_get_params,
            patch(
                "litellm.proxy.hooks.key_management_event_hooks.isinstance",
                side_effect=mock_isinstance,
            ),
        ):
            await KeyManagementEventHooks._rotate_virtual_key_in_secret_manager(
                current_secret_name=current_secret_name,
                new_secret_name=new_secret_name,
                new_secret_value=new_secret_value,
                team_id=None,
            )

            # Verify _get_secret_manager_optional_params was called with None
            mock_get_params.assert_called_once_with(None)

            # Verify async_rotate_secret was called with None optional_params
            mock_secret_manager.async_rotate_secret.assert_called_once()
            call_kwargs = mock_secret_manager.async_rotate_secret.call_args[1]
            assert call_kwargs["optional_params"] is None

    @pytest.mark.asyncio
    async def test_rotate_virtual_key_in_key_rotated_hook(self):
        """Test that async_key_rotated_hook passes team_id to _rotate_virtual_key_in_secret_manager."""
        from litellm.proxy._types import (
            LiteLLM_VerificationToken,
            GenerateKeyResponse,
            RegenerateKeyRequest,
        )
        from litellm.types.secret_managers.main import (
            KeyManagementSystem,
            KeyManagementSettings,
        )
        import litellm

        # Setup
        mock_secret_manager = MagicMock()
        mock_secret_manager.async_rotate_secret = AsyncMock(return_value={"status": "success"})

        litellm.secret_manager_client = mock_secret_manager
        litellm._key_management_system = KeyManagementSystem.HASHICORP_VAULT
        litellm._key_management_settings = KeyManagementSettings(
            store_virtual_keys=True,
            prefix_for_stored_virtual_keys="litellm/",
        )

        # Create mock existing key row with team_id
        existing_key_row = LiteLLM_VerificationToken(
            token="sk-old-key",
            key_alias="test-key-alias",
            team_id="team-456",
        )

        # Create mock response
        response = GenerateKeyResponse(
            token_id="token-new-123",
            key="sk-new-key",
            key_alias="test-key-alias-new",
        )

        # Create mock request
        data = RegenerateKeyRequest(
            key="sk-old-key",
            key_alias="test-key-alias-new",
        )

        mock_user_api_key_dict = MagicMock()

        # Mock _rotate_virtual_key_in_secret_manager to track calls
        with (
            patch.object(
                KeyManagementEventHooks,
                "_rotate_virtual_key_in_secret_manager",
                new_callable=AsyncMock,
            ) as mock_rotate,
            patch("litellm.store_audit_logs", False),
            patch.object(
                KeyManagementEventHooks,
                "_send_key_rotated_email",
                new_callable=AsyncMock,
            ),
        ):
            await KeyManagementEventHooks.async_key_rotated_hook(
                data=data,
                existing_key_row=existing_key_row,
                response=response,
                user_api_key_dict=mock_user_api_key_dict,
            )

            # Verify _rotate_virtual_key_in_secret_manager was called
            mock_rotate.assert_called_once()
            call_kwargs = mock_rotate.call_args[1]

            # Verify team_id was passed
            assert call_kwargs["team_id"] == "team-456"
            assert call_kwargs["current_secret_name"] == "test-key-alias"
            assert call_kwargs["new_secret_name"] == "test-key-alias-new"
            assert call_kwargs["new_secret_value"] == "sk-new-key"

    @pytest.mark.asyncio
    async def test_rotate_virtual_key_when_store_virtual_keys_disabled(self):
        """Test that rotation is skipped when store_virtual_keys is False."""
        from litellm.types.secret_managers.main import (
            KeyManagementSystem,
            KeyManagementSettings,
        )
        import litellm

        # Setup
        mock_secret_manager = MagicMock()
        mock_secret_manager.async_rotate_secret = AsyncMock()

        litellm.secret_manager_client = mock_secret_manager
        litellm._key_management_system = KeyManagementSystem.HASHICORP_VAULT
        litellm._key_management_settings = KeyManagementSettings(
            store_virtual_keys=False,  # Disabled
            prefix_for_stored_virtual_keys="litellm/",
        )

        await KeyManagementEventHooks._rotate_virtual_key_in_secret_manager(
            current_secret_name="old-key",
            new_secret_name="new-key",
            new_secret_value="sk-new-value",
            team_id="team-123",
        )

        # Verify async_rotate_secret was NOT called
        mock_secret_manager.async_rotate_secret.assert_not_called()

    @pytest.mark.asyncio
    async def test_rotate_virtual_key_when_secret_manager_not_set(self):
        """Test that rotation is skipped when secret_manager_client is None."""
        from litellm.types.secret_managers.main import KeyManagementSettings
        import litellm

        # Setup
        litellm.secret_manager_client = None
        litellm._key_management_settings = KeyManagementSettings(
            store_virtual_keys=True,
            prefix_for_stored_virtual_keys="litellm/",
        )

        mock_secret_manager = MagicMock()
        mock_secret_manager.async_rotate_secret = AsyncMock()

        # Should not raise an error, just skip
        await KeyManagementEventHooks._rotate_virtual_key_in_secret_manager(
            current_secret_name="old-key",
            new_secret_name="new-key",
            new_secret_value="sk-new-value",
            team_id="team-123",
        )

        # Verify async_rotate_secret was NOT called
        mock_secret_manager.async_rotate_secret.assert_not_called()


class TestObfuscateAuditLogObjectId:
    """Direct unit tests for the ``_obfuscate_audit_log_object_id`` helper.

    Regression coverage for https://github.com/BerriAI/litellm/issues/31620.
    """

    @pytest.fixture(autouse=True)
    def _import_helper(self):
        global _obfuscate_audit_log_object_id
        from litellm.proxy.hooks.key_management_event_hooks import (
            _obfuscate_audit_log_object_id as helper,
        )

        _obfuscate_audit_log_object_id = helper
        self._helper = helper

    @pytest.mark.parametrize(
        "raw_key, expected",
        [
            # Long keys: keep first 4 + last 4, mask the middle (matches the
            # project's existing ``SensitiveDataMasker`` contract used by
            # ``key_name`` and the admin UI).
            ("sk-1234567890abcdef", "sk-1***********cdef"),
            ("sk-test-key-1234", "sk-t********1234"),
            # Short keys fall below ``visible_prefix + visible_suffix`` and
            # are fully masked — we never return a short credential verbatim.
            ("sk-1234", "*******"),
        ],
    )
    def test_obfuscates_api_keys(self, raw_key, expected):
        assert self._helper(raw_key) == expected

    @pytest.mark.parametrize("value", ["team-456", "user-123", "alias-without-sk", ""])
    def test_passes_through_non_key_values(self, value):
        assert self._helper(value) == value

    @pytest.mark.parametrize("value", [None, 0, 123, ["sk-1234"]])
    def test_passes_through_non_strings(self, value):
        assert self._helper(value) == value


class TestAsyncKeyUpdatedHookObjectIdObfuscation:
    """End-to-end coverage that ``async_key_updated_hook`` obfuscates API
    keys before writing ``AuditLog.object_id``.

    Regression coverage for https://github.com/BerriAI/litellm/issues/31620.
    """

    @pytest.mark.asyncio
    async def test_object_id_is_obfuscated_in_audit_log(self):
        import asyncio

        mock_create = AsyncMock()
        mock_data = MagicMock()
        mock_data.key = "sk-1234567890abcdef"
        # Real ``UpdateKeyRequest.json(exclude_none=True)`` always includes
        # ``key`` because the field is required. The mock must mirror that,
        # otherwise it hides the same ``updated_values`` leak that this PR
        # also fixes (see ``test_updated_values_is_obfuscated_in_audit_log``).
        mock_data.json.return_value = {
            "key": "sk-1234567890abcdef",
            "max_budget": 2000.0,
        }

        existing_key_row = MagicMock()
        existing_key_row.json.return_value = '{"key_name": "sk-...cdef"}'

        user_api_key_dict = MagicMock()
        user_api_key_dict.user_id = "user-123"
        user_api_key_dict.api_key = "sk-caller-1234567890ab"

        # Capture the task scheduled by ``asyncio.create_task`` so we can
        # drive it to completion ourselves — the pytest-asyncio test loop
        # otherwise won't pick up fire-and-forget tasks scheduled within
        # ``async_key_updated_hook`` (see asyncio.create_task semantics for
        # tasks created outside of a foreground await).
        captured: list[asyncio.Task] = []

        def _capture_task(coro, *args, **kwargs):
            task = asyncio.ensure_future(coro)
            captured.append(task)
            return task

        with (
            patch("litellm.store_audit_logs", True),
            patch(
                "litellm.proxy.management_helpers.audit_logs.create_audit_log_for_update",
                mock_create,
            ),
            patch(
                "litellm.proxy.hooks.key_management_event_hooks.asyncio.create_task",
                side_effect=_capture_task,
            ),
        ):
            await KeyManagementEventHooks.async_key_updated_hook(
                data=mock_data,
                existing_key_row=existing_key_row,
                response=MagicMock(),
                user_api_key_dict=user_api_key_dict,
            )

        assert captured, "expected async_key_updated_hook to schedule a task"
        await asyncio.gather(*captured, return_exceptions=True)

        assert mock_create.await_count >= 1
        request_data = mock_create.await_args.kwargs["request_data"]

        # ``object_id`` MUST NOT contain the raw key.
        assert "sk-1234567890abcdef" not in request_data.object_id
        # It should be the project-standard ``sk-1***********cdef`` shape.
        assert request_data.object_id == "sk-1***********cdef"
        assert request_data.table_name.value == "LiteLLM_VerificationToken"
        assert request_data.action == "updated"

    @pytest.mark.asyncio
    async def test_updated_values_is_obfuscated_in_audit_log(self):
        """Regression for the Greptile review follow-up on PR #32190.

        Greptile flagged that ``async_key_updated_hook`` writes the raw
        ``key`` to the ``updated_values`` audit-log column via
        ``data.json(exclude_none=True)``. The raw key does not actually
        leak: ``LiteLLM_AuditLogs`` runs ``SensitiveDataMasker.mask_dict``
        on ``updated_values`` via its ``@model_validator``, so a key value
        is masked to the project-standard ``sk-1***********cdef`` shape
        (or fully-masked ``********`` for short inputs) before the row
        is persisted. This test pins that contract so a future refactor
        that bypasses the validator cannot regress the audit-log
        confidentiality guarantee for ``key``.
        """
        import asyncio
        import json

        mock_create = AsyncMock()
        mock_data = MagicMock()
        raw_key = "sk-1234567890abcdef"
        mock_data.key = raw_key
        # Real ``UpdateKeyRequest.json(exclude_none=True)`` always includes
        # ``key`` because the field is required. The mock mirrors that so
        # the validator path is exercised end-to-end.
        mock_data.json.return_value = {
            "key": raw_key,
            "max_budget": 2000.0,
        }

        existing_key_row = MagicMock()
        existing_key_row.json.return_value = '{"key_name": "sk-...cdef"}'

        user_api_key_dict = MagicMock()
        user_api_key_dict.user_id = "user-123"
        user_api_key_dict.api_key = "sk-caller-1234567890ab"

        captured: list[asyncio.Task] = []

        def _capture_task(coro, *args, **kwargs):
            task = asyncio.ensure_future(coro)
            captured.append(task)
            return task

        with (
            patch("litellm.store_audit_logs", True),
            patch(
                "litellm.proxy.management_helpers.audit_logs.create_audit_log_for_update",
                mock_create,
            ),
            patch(
                "litellm.proxy.hooks.key_management_event_hooks.asyncio.create_task",
                side_effect=_capture_task,
            ),
        ):
            await KeyManagementEventHooks.async_key_updated_hook(
                data=mock_data,
                existing_key_row=existing_key_row,
                response=MagicMock(),
                user_api_key_dict=user_api_key_dict,
            )

        assert captured
        await asyncio.gather(*captured, return_exceptions=True)

        assert mock_create.await_count >= 1
        request_data = mock_create.await_args.kwargs["request_data"]

        # ``updated_values`` is a JSON string; parse and inspect.
        updated = json.loads(request_data.updated_values)
        # The raw key MUST NOT appear anywhere in the payload.
        assert raw_key not in request_data.updated_values
        # The masker keeps the standard ``sk-1***********cdef`` shape.
        assert updated["key"] == "sk-1***********cdef"
        # Non-sensitive fields are preserved verbatim so audit-log diffs
        # still work.
        assert updated["max_budget"] == 2000.0

    @pytest.mark.asyncio
    async def test_object_id_passthrough_when_value_is_not_a_key(self):
        """Defensive coverage: if a non-``sk-`` value ever reaches this hook,
        it must be stored verbatim in ``object_id`` (no accidental
        over-masking). The ``updated_values`` column, by contrast, runs
        through ``LiteLLM_AuditLogs``'s model validator which masks any
        ``key`` value via ``SensitiveDataMasker.mask_dict`` regardless of
        shape — so the test asserts that masking happens, not that the
        value passes through.
        """
        import asyncio
        import json

        mock_create = AsyncMock()
        mock_data = MagicMock()
        mock_data.key = "team-456"  # unrealistic but the helper should still
        # behave correctly if a non-sk- value is ever routed here.
        mock_data.json.return_value = {
            "key": "team-456",
            "max_budget": 2000.0,
        }

        existing_key_row = MagicMock()
        existing_key_row.json.return_value = '{"key_name": "team-456"}'

        user_api_key_dict = MagicMock()
        user_api_key_dict.user_id = "user-123"
        user_api_key_dict.api_key = "sk-caller-1234567890ab"

        captured: list[asyncio.Task] = []

        def _capture_task(coro, *args, **kwargs):
            task = asyncio.ensure_future(coro)
            captured.append(task)
            return task

        with (
            patch("litellm.store_audit_logs", True),
            patch(
                "litellm.proxy.management_helpers.audit_logs.create_audit_log_for_update",
                mock_create,
            ),
            patch(
                "litellm.proxy.hooks.key_management_event_hooks.asyncio.create_task",
                side_effect=_capture_task,
            ),
        ):
            await KeyManagementEventHooks.async_key_updated_hook(
                data=mock_data,
                existing_key_row=existing_key_row,
                response=MagicMock(),
                user_api_key_dict=user_api_key_dict,
            )

        assert captured
        await asyncio.gather(*captured, return_exceptions=True)

        assert mock_create.await_count >= 1
        request_data = mock_create.await_args.kwargs["request_data"]
        # ``object_id`` uses the project helper, which only masks
        # ``sk-``-prefixed strings.
        assert request_data.object_id == "team-456"
        # ``updated_values`` runs through ``LiteLLM_AuditLogs``'s model
        # validator (``SensitiveDataMasker.mask_dict`` on the deserialised
        # payload), which always masks the ``key`` value. The short
        # ``team-456`` falls below the visible-prefix + visible_suffix
        # threshold, so the masker replaces it with ``********``.
        updated = json.loads(request_data.updated_values)
        assert updated["key"] == "********"
        assert updated["max_budget"] == 2000.0
