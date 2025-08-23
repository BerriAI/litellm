import os
import sys

import pytest

# Ensure repository root is on sys.path, matching nearby tests' pattern
sys.path.insert(0, os.path.abspath("../../../.."))

from pydantic import ValidationError

from litellm.types.proxy.management_endpoints.internal_user_endpoints import (
    BulkUpdateUserRequest,
)


def test_bulk_update_request_empty_payload_validation_error():
    """Explicit empty user_updates should fail validation with a helpful message."""
    with pytest.raises((ValidationError, ValueError)) as exc:
        BulkUpdateUserRequest(user_updates=None)
    assert (
        "Must specify either 'users' for individual updates or 'all_users=True' with 'user_updates' for bulk updates"
        in str(exc.value)
    )


def test_bulk_update_request_both_users_and_all_users_error():
    """Specifying both 'users' and 'all_users=True' should raise a validation error when user_updates is present."""
    with pytest.raises((ValidationError, ValueError)) as exc:
        BulkUpdateUserRequest(
            users=[{"user_id": "user1", "user_role": "internal_user"}],
            all_users=True,
            user_updates={"user_role": "internal_user"},
        )
    assert "Cannot specify both 'users' and 'all_users=True'" in str(exc.value)


def test_bulk_update_request_all_users_requires_user_updates():
    """all_users=True with explicit user_updates=None should fail validation."""
    with pytest.raises((ValidationError, ValueError)) as exc:
        BulkUpdateUserRequest(all_users=True, user_updates=None)
    assert (
        "Must specify either 'users' for individual updates or 'all_users=True' with 'user_updates' for bulk updates"
        in str(exc.value)
    )


def test_bulk_update_request_all_users_with_updates_valid():
    """A valid all_users request with user_updates parses successfully."""
    req = BulkUpdateUserRequest(
        all_users=True,
        user_updates={
            "user_role": "internal_user",
            "max_budget": 50.0,
        },
    )
    assert req.all_users is True
    assert req.user_updates is not None

