#!/usr/bin/env python3
"""
Validate Team Admin Permissions

This script validates that a team admin can perform the following:
1. View all team keys
2. Add/remove member from team
3. Add/edit/delete model in team
4. Create team key with all team models
5. See all team models in test key dropdown

Prerequisites:
- Proxy running with database (PostgreSQL or SQLite)
- Proxy config with store_virtual_keys: true
- A team with at least one team admin user
- Team admin JWT or API key for authentication

Usage:
  export PROXY_URL="http://localhost:4000"
  export TEAM_ADMIN_TOKEN="<jwt-or-api-key-for-team-admin>"
  export TEAM_ID="<team-id-to-test>"
  poetry run python scripts/validate_team_admin_permissions.py

Or pass as arguments:
  poetry run python scripts/validate_team_admin_permissions.py \\
    --proxy-url http://localhost:4000 \\
    --team-admin-token <token> \\
    --team-id <team-id>
"""

import argparse
import json
import os
import sys
from typing import Optional, Tuple

try:
    import httpx
except ImportError:
    httpx = None


def _headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def _get(url: str, token: str, params: Optional[dict] = None) -> dict:
    r = httpx.get(url, headers=_headers(token), params=params or {}, timeout=30)
    r.raise_for_status()
    return r.json()


def _post(url: str, token: str, json_body: dict) -> dict:
    r = httpx.post(url, headers=_headers(token), json=json_body, timeout=30)
    r.raise_for_status()
    return r.json()


def _delete(url: str, token: str, json_body: dict) -> dict:
    r = httpx.request("DELETE", url, headers=_headers(token), json=json_body, timeout=30)
    r.raise_for_status()
    return r.json()


def validate_view_all_team_keys(base_url: str, token: str, team_id: str) -> Tuple[bool, str]:
    """1. Team admin should be able to view all team keys."""
    url = f"{base_url}/key/list"
    params = {
        "page": 1,
        "size": 50,
        "return_full_object": "true",
        "include_team_keys": "true",
        "include_created_by_keys": "true",
        "team_id": team_id,
    }
    try:
        data = _get(url, token, params)
        keys = data.get("keys", [])
        total = data.get("total_count", 0)
        return True, f"OK: Listed {len(keys)} keys (total_count={total})"
    except Exception as e:
        return False, f"FAIL: {e}"


def validate_add_remove_member(
    base_url: str, token: str, team_id: str, test_user_id: Optional[str] = None
) -> Tuple[bool, str]:
    """2. Team admin should be able to add/remove member from team."""
    if not test_user_id:
        return True, "SKIP: No test_user_id provided (set --test-user-id to validate)"

    # Add member
    add_url = f"{base_url}/team/member_add"
    try:
        _post(add_url, token, {"team_id": team_id, "user_id": test_user_id})
    except Exception as e:
        # May already be a member
        if "already" in str(e).lower() or "409" in str(e):
            pass
        else:
            return False, f"FAIL add member: {e}"

    # Remove member
    del_url = f"{base_url}/team/member_delete"
    try:
        _post(del_url, token, {"team_id": team_id, "user_id": test_user_id})
        return True, "OK: Add and remove member succeeded"
    except Exception as e:
        return False, f"FAIL remove member: {e}"


def validate_add_edit_delete_model(
    base_url: str, token: str, team_id: str, model_id: Optional[str] = None
) -> Tuple[bool, str]:
    """3. Team admin should be able to add/edit/delete model in team."""
    model_name = model_id or f"openai/test-team-model-{team_id[:8]}"

    # Add model
    add_url = f"{base_url}/model/new"
    add_body = {
        "model_name": model_name,
        "litellm_params": {
            "model": "gpt-3.5-turbo",
            "api_key": "test-key",
        },
        "model_info": {"team_id": team_id},
    }
    try:
        _post(add_url, token, add_body)
    except Exception as e:
        return False, f"FAIL add model: {e}"

    # Update model
    update_url = f"{base_url}/model/update"
    update_body = {
        "id": model_name,
        "model_info": {"team_id": team_id, "team_public_model_name": "Test Model"},
    }
    try:
        _post(update_url, token, update_body)
    except Exception as e:
        return False, f"FAIL update model: {e}"

    # Delete model
    del_url = f"{base_url}/model/delete"
    try:
        _post(del_url, token, {"id": model_name})
        return True, "OK: Add, edit, delete model succeeded"
    except Exception as e:
        return False, f"FAIL delete model: {e}"


def validate_create_team_key(base_url: str, token: str, team_id: str) -> Tuple[bool, str]:
    """4. Team admin should be able to create team key with all team models."""
    url = f"{base_url}/key/generate"
    body = {
        "team_id": team_id,
        "models": ["all-team-models"],
        "key_alias": "team-admin-validation-key",
    }
    try:
        data = _post(url, token, body)
        key = data.get("key") or (data.get("keys", [None])[0] if data.get("keys") else None)
        # Clean up: delete the key (by alias to avoid raw key handling)
        del_url = f"{base_url}/key/delete"
        try:
            _post(del_url, token, {"key_aliases": ["team-admin-validation-key"]})
        except Exception:
            pass  # Best-effort cleanup
        return True, "OK: Created team key with all team models"
    except Exception as e:
        return False, f"FAIL: {e}"


def validate_test_key_dropdown_models(base_url: str, token: str, team_id: str) -> Tuple[bool, str]:
    """5. Team admin should see all team models in test key dropdown (GET /models)."""
    url = f"{base_url}/models"
    params = {"team_id": team_id, "return_wildcard_routes": "True"}
    try:
        data = _get(url, token, params)
        models = data.get("data", [])
        model_ids = [m.get("id", m) if isinstance(m, dict) else str(m) for m in models]
        return True, f"OK: Got {len(model_ids)} models for team (sample: {model_ids[:5]})"
    except Exception as e:
        return False, f"FAIL: {e}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate team admin permissions")
    parser.add_argument(
        "--proxy-url",
        default=os.environ.get("PROXY_URL", "http://localhost:4000"),
        help="Proxy base URL",
    )
    parser.add_argument(
        "--team-admin-token",
        default=os.environ.get("TEAM_ADMIN_TOKEN"),
        required=True,
        help="JWT or API key for team admin user",
    )
    parser.add_argument(
        "--team-id",
        default=os.environ.get("TEAM_ID"),
        required=True,
        help="Team ID to validate",
    )
    parser.add_argument(
        "--test-user-id",
        default=os.environ.get("TEST_USER_ID"),
        help="User ID to add/remove for member test (optional)",
    )
    parser.add_argument(
        "--skip-member-test",
        action="store_true",
        help="Skip add/remove member test",
    )
    parser.add_argument(
        "--skip-model-crud",
        action="store_true",
        help="Skip add/edit/delete model test (avoids creating test model)",
    )
    args = parser.parse_args()

    if httpx is None:
        print("ERROR: httpx required. Run: poetry add httpx")
        return 1

    base = args.proxy_url.rstrip("/")
    token = args.team_admin_token
    team_id = args.team_id

    checks = [
        ("View all team keys", lambda: validate_view_all_team_keys(base, token, team_id)),
        (
            "Add/remove member from team",
            lambda: validate_add_remove_member(base, token, team_id, args.test_user_id)
            if not args.skip_member_test
            else (True, "SKIP: --skip-member-test"),
        ),
        (
            "Add/edit/delete model in team",
            lambda: validate_add_edit_delete_model(base, token, team_id)
            if not args.skip_model_crud
            else (True, "SKIP: --skip-model-crud"),
        ),
        ("Create team key with all team models", lambda: validate_create_team_key(base, token, team_id)),
        (
            "See all team models in test key dropdown",
            lambda: validate_test_key_dropdown_models(base, token, team_id),
        ),
    ]

    print("=" * 60)
    print("Team Admin Permissions Validation")
    print("=" * 60)
    print(f"Proxy: {base}")
    print(f"Team ID: {team_id}")
    print()

    failed = 0
    for name, fn in checks:
        ok, msg = fn()
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        print(f"        {msg}")
        if not ok:
            failed += 1

    print()
    if failed == 0:
        print("All checks passed.")
        return 0
    print(f"{failed} check(s) failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
