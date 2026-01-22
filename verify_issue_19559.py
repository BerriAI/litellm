#!/usr/bin/env python3
"""
Verification script for Issue #19559: Passwords with non-ASCII characters

This script verifies that login works with non-ASCII characters (like £) in passwords.
It tests both the bug scenario and the fix.

Usage:
    poetry run python verify_issue_19559.py
"""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

# Test 1: Direct comparison test (basic proof of concept)
def test_secrets_compare_with_nonascii():
    """Test that secrets.compare_digest works with UTF-8 encoded non-ASCII strings"""
    import secrets

    print("\n" + "="*70)
    print("TEST 1: secrets.compare_digest with non-ASCII characters")
    print("="*70)

    test_password = "test£pass"

    # Show what fails
    print(f"\nPassword with £ symbol: {test_password!r}")
    print("\nAttempt 1: Direct comparison (FAILS)")
    try:
        result = secrets.compare_digest(test_password, test_password)
        print(f"  ✗ Should have failed but got: {result}")
        return False
    except TypeError as e:
        print(f"  ✓ Expected error: {e}")

    # Show what works
    print("\nAttempt 2: UTF-8 encoded comparison (WORKS - THE FIX)")
    try:
        result = secrets.compare_digest(
            test_password.encode("utf-8"),
            test_password.encode("utf-8")
        )
        print(f"  ✓ Comparison succeeded: {result}")
        return True
    except TypeError as e:
        print(f"  ✗ Unexpected error: {e}")
        return False


# Test 2: Actual authenticate_user function with non-ASCII
async def test_authenticate_user_with_nonascii():
    """Test the actual authenticate_user function with non-ASCII password"""
    from litellm.proxy.auth.login_utils import authenticate_user
    from litellm.proxy._types import ProxyException

    print("\n" + "="*70)
    print("TEST 2: authenticate_user function with non-ASCII password")
    print("="*70)

    password = "admin£password123"
    username = "admin"

    print(f"\nScenario: User with password containing £ symbol")
    print(f"  Username: {username!r}")
    print(f"  Password: {password!r}")

    try:
        mock_prisma = MagicMock()
        mock_prisma.db.litellm_usertable.find_first = AsyncMock(return_value=None)

        with patch.dict(os.environ, {
            "DATABASE_URL": "postgresql://test:test@localhost/test",
            "UI_USERNAME": username,
            "UI_PASSWORD": password
        }):
            with patch("litellm.proxy.auth.login_utils.generate_key_helper_fn", new_callable=AsyncMock) as mock_gen:
                mock_gen.return_value = {"token": "test-token", "user_id": "test-admin"}
                with patch("litellm.proxy.auth.login_utils.user_update", new_callable=AsyncMock):
                    with patch("litellm.proxy.auth.login_utils.get_secret_bool", return_value=False):
                        result = await authenticate_user(
                            username=username,
                            password=password,
                            master_key="sk-1234",
                            prisma_client=mock_prisma,
                        )

        print(f"\n✓ LOGIN SUCCESSFUL")
        print(f"  User ID: {result.user_id}")
        print(f"  User Role: {result.user_role}")
        return True

    except TypeError as e:
        if "comparing strings with non-ASCII" in str(e):
            print(f"\n✗ LOGIN FAILED (BUG #19559)")
            print(f"  Error: {e}")
            print(f"\n  This is the exact error from the bug report!")
            return False
        raise
    except ProxyException as e:
        print(f"\n✗ ProxyException: {e.message}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


# Test 3: Database user with non-ASCII password
async def test_db_user_with_nonascii():
    """Test database user login with non-ASCII password"""
    from litellm.proxy.auth.login_utils import authenticate_user
    from litellm.proxy._types import hash_token

    print("\n" + "="*70)
    print("TEST 3: Database user login with non-ASCII password")
    print("="*70)

    user_email = "user@example.com"
    password = "secure£pass£word"
    hashed_password = hash_token(token=password)

    print(f"\nScenario: Database user with non-ASCII password")
    print(f"  Email: {user_email!r}")
    print(f"  Password: {password!r}")

    try:
        mock_user = MagicMock()
        mock_user.user_id = "user-123"
        mock_user.user_email = user_email
        mock_user.password = hashed_password
        mock_user.user_role = "internal_user"

        def mock_find_first(**kwargs):
            where = kwargs.get("where", {})
            if user_email.lower() in str(where).lower():
                return mock_user
            return None

        mock_prisma = MagicMock()
        mock_prisma.db.litellm_usertable.find_first = AsyncMock(side_effect=mock_find_first)

        with patch.dict(os.environ, {
            "DATABASE_URL": "postgresql://test:test@localhost/test",
            "UI_USERNAME": "admin",
            "UI_PASSWORD": "admin-pass"
        }):
            with patch("litellm.proxy.auth.login_utils.expire_previous_ui_session_tokens", new_callable=AsyncMock):
                with patch("litellm.proxy.auth.login_utils.generate_key_helper_fn", new_callable=AsyncMock) as mock_gen:
                    mock_gen.return_value = {"token": "db-token"}

                    result = await authenticate_user(
                        username=user_email,
                        password=password,
                        master_key="sk-1234",
                        prisma_client=mock_prisma,
                    )

        print(f"\n✓ LOGIN SUCCESSFUL")
        print(f"  User ID: {result.user_id}")
        print(f"  User Email: {result.user_email}")
        return True

    except TypeError as e:
        if "comparing strings with non-ASCII" in str(e):
            print(f"\n✗ LOGIN FAILED (BUG #19559)")
            print(f"  Error: {e}")
            return False
        raise
    except Exception as e:
        print(f"\n✗ Error: {type(e).__name__}: {e}")
        return False


async def main():
    """Run all verification tests"""
    print("\n" + "█"*70)
    print("█ VERIFICATION SCRIPT FOR ISSUE #19559")
    print("█ Passwords with non-ASCII characters (£ symbol)")
    print("█"*70)

    results = []

    # Test 1: Basic secrets comparison
    results.append(("Direct comparison with UTF-8 encoding", test_secrets_compare_with_nonascii()))

    # Test 2: Admin login with non-ASCII
    results.append(("Admin login with non-ASCII password", await test_authenticate_user_with_nonascii()))

    # Test 3: DB user with non-ASCII
    results.append(("Database user with non-ASCII password", await test_db_user_with_nonascii()))

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(result[1] for result in results)

    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL TESTS PASSED - Issue #19559 is fixed!")
    else:
        print("✗ SOME TESTS FAILED - Issue #19559 may not be fixed")
    print("="*70 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
