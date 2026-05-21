#!/usr/bin/env python3
"""
End-to-end HTTP test: mock OIDC + running LiteLLM proxy + CLI SSO claim map.

Prerequisites:
  1. Postgres (or your DB) reachable via DATABASE_URL
  2. Mock IdP:  python scripts/mock_oidc_server_for_cli_sso.py
  3. Proxy with Generic SSO env (restart proxy after setting these):

     export PROXY_BASE_URL=http://127.0.0.1:4000/
     export CLI_SSO_CLAIM_MAP="employment_type->acme_employment_type,org_info.department->department"
     export GENERIC_CLIENT_ID=litellm-cli-test
     export GENERIC_CLIENT_SECRET=litellm-cli-test-secret
     export GENERIC_AUTHORIZATION_ENDPOINT=http://127.0.0.1:8765/authorize
     export GENERIC_TOKEN_ENDPOINT=http://127.0.0.1:8765/token
     export GENERIC_USERINFO_ENDPOINT=http://127.0.0.1:8765/userinfo
     export GENERIC_USER_EXTRA_ATTRIBUTES=employment_type,org_info.department

     poetry run litellm --config proxy_server_config.yaml --port 4000

  4. Run this script:

     python scripts/test_cli_sso_claims_e2e.py

Alternative without HTTP/SSO: python manual_cli_sso_claims.py
"""

from __future__ import annotations

import os
import re
import sys

import requests

PROXY_BASE = os.getenv("PROXY_BASE_URL", "http://127.0.0.1:4000").rstrip("/")
CLI_SOURCE = "litellm-cli"


def _extract_hidden_input(html: str, name: str) -> str:
    match = re.search(
        rf'<input[^>]+name="{re.escape(name)}"[^>]+value="([^"]*)"',
        html,
    )
    if not match:
        raise RuntimeError(f"Could not find hidden input {name!r} in HTML")
    return match.group(1)


def main() -> int:
    session = requests.Session()

    print("1) POST /sso/cli/start")
    start = session.post(f"{PROXY_BASE}/sso/cli/start", timeout=30)
    start.raise_for_status()
    flow = start.json()
    login_id = flow["login_id"]
    poll_secret = flow["poll_secret"]
    user_code = flow["user_code"]
    print(f"   login_id={login_id}  user_code={user_code}")

    print("2) GET /sso/key/generate (follow redirects through mock IdP)")
    generate_url = (
        f"{PROXY_BASE}/sso/key/generate" f"?source={CLI_SOURCE}&key={login_id}"
    )
    browser = session.get(generate_url, timeout=30, allow_redirects=True)
    if not browser.ok:
        print(f"   HTTP {browser.status_code} from {browser.url}")
        print(f"   Body: {browser.text[:2000]}")
        browser.raise_for_status()
    if "Complete CLI Login" not in browser.text:
        print("   Expected verification page HTML; got:", browser.text[:500])
        return 1
    browser_complete_token = _extract_hidden_input(
        browser.text, "browser_complete_token"
    )

    print("3) POST /sso/cli/complete/{login_id}")
    complete = session.post(
        f"{PROXY_BASE}/sso/cli/complete/{login_id}",
        data={
            "user_code": user_code,
            "browser_complete_token": browser_complete_token,
        },
        timeout=30,
    )
    complete.raise_for_status()

    print("4) GET /sso/cli/poll/{login_id}")
    poll = requests.get(
        f"{PROXY_BASE}/sso/cli/poll/{login_id}",
        headers={"x-litellm-cli-poll-secret": poll_secret},
        timeout=30,
    )
    poll.raise_for_status()
    data = poll.json()
    print("   Poll response:", data)

    expected = {
        "acme_employment_type": os.getenv("MOCK_OIDC_EMPLOYMENT_TYPE", "contractor"),
        "department": os.getenv("MOCK_OIDC_DEPARTMENT", "Engineering"),
    }
    attribution = data.get("attribution_metadata") or {}
    if data.get("status") != "ready" or not data.get("key"):
        print("FAIL: expected status=ready with JWT key")
        return 1
    for key, value in expected.items():
        if attribution.get(key) != value:
            print(
                f"FAIL: attribution_metadata[{key!r}] = {attribution.get(key)!r}, want {value!r}"
            )
            return 1

    print("OK: CLI SSO claim map surfaced in poll response")
    return 0


if __name__ == "__main__":
    sys.exit(main())
