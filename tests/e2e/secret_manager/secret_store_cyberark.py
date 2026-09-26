from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass, field
from typing import Final, Literal
from urllib.parse import quote

import pytest
import yaml
from e2e_http import ExternalWrite, Headers, send_text_external
from pydantic import Field

from secret_store import SecretBackend

CYBERARK_API_BASE_ENV: Final = "E2E_CYBERARK_API_BASE"
CYBERARK_ACCOUNT_ENV: Final = "E2E_CYBERARK_ACCOUNT"
CYBERARK_USERNAME_ENV: Final = "E2E_CYBERARK_USERNAME"
CYBERARK_API_KEY_ENV: Final = "E2E_CYBERARK_API_KEY"

# The same defaults CyberArkSecretManager falls back to for CYBERARK_*.
DEFAULT_API_BASE: Final = "http://127.0.0.1:8080"
DEFAULT_ACCOUNT: Final = "default"
DEFAULT_USERNAME: Final = "admin"

SYSTEM: Final = "cyberark"

_POLICY_LOAD_ATTEMPTS: Final = 5
_POLICY_LOAD_RETRY_DELAY_SECONDS: Final = 0.2

_START_HINT: Final = (
    f"Start one with `bash tests/e2e/secret_manager/backend.sh up {SYSTEM}`, which writes the env for "
    f"the proxy (booted from gateway/secret_manager_{SYSTEM}_ci_config.yml) and for the tests"
)


class ConjurHeaders(Headers):
    authorization: str = Field(repr=False)
    content_type: str | None = Field(default=None, serialization_alias="Content-Type")


def _policy_scalar(name: str) -> str:
    # Quoted the way CyberArkSecretManager._ensure_variable_exists quotes it.
    return yaml.safe_dump(name, default_style='"').strip()


@dataclass(frozen=True, slots=True)
class Conjur:
    base_url: str
    account: str
    username: str
    api_key: str = field(repr=False)

    def _fail_unless_reached(self, result: ExternalWrite, action: str) -> None:
        if result.status_code == -1:
            pytest.fail(f"No live Conjur at {self.base_url}: {result.body}. {_START_HINT}")
        if result.status_code == 401:
            pytest.fail(f"Conjur rejected {self.username}'s credentials while trying to {action}. {_START_HINT}")

    def _headers(self, content_type: str | None = None) -> ConjurHeaders:
        # Tokens last about eight minutes, so each call authenticates afresh rather than
        # letting a long session outlive a cached one.
        auth: Final = send_text_external(
            "POST",
            f"{self.base_url}/authn/{self.account}/{quote(self.username, safe='')}/authenticate",
            headers=Headers(),
            content=self.api_key,
        )
        self._fail_unless_reached(auth, "authenticate")
        if not auth.ok:
            pytest.fail(f"Conjur refused to authenticate {self.username}: HTTP {auth.status_code} {auth.body[:300]}")
        token: Final = base64.b64encode(auth.body.encode()).decode()
        return ConjurHeaders(authorization=f'Token token="{token}"', content_type=content_type)

    def _secret_url(self, name: str) -> str:
        return f"{self.base_url}/secrets/{self.account}/variable/{quote(name, safe='')}"

    def _load_root_policy(self, method: Literal["POST", "PATCH"], policy: str, attempt: int = 0) -> ExternalWrite:
        result: Final = send_text_external(
            method,
            f"{self.base_url}/policies/{self.account}/policy/root",
            headers=self._headers(content_type="application/x-yaml"),
            content=policy,
        )
        if result.status_code != 409 or attempt + 1 == _POLICY_LOAD_ATTEMPTS:
            return result
        time.sleep(_POLICY_LOAD_RETRY_DELAY_SECONDS * 2**attempt)
        return self._load_root_policy(method, policy, attempt + 1)

    def _update_root_policy(self, method: Literal["POST", "PATCH"], policy: str, action: str) -> None:
        result: Final = self._load_root_policy(method, policy)
        self._fail_unless_reached(result, action)
        if not result.ok:
            pytest.fail(f"Conjur refused to {action}: HTTP {result.status_code} {result.body[:300]}")

    def write(self, name: str, value: str) -> None:
        self._update_root_policy("POST", f"- !variable {_policy_scalar(name)}\n", f"declare {name}")
        result: Final = send_text_external("POST", self._secret_url(name), headers=self._headers(), content=value)
        self._fail_unless_reached(result, f"write {name}")
        if not result.ok:
            pytest.fail(f"Conjur refused to write {name}: HTTP {result.status_code} {result.body[:300]}")

    def read(self, name: str) -> str | None:
        result: Final = send_text_external("GET", self._secret_url(name), headers=self._headers())
        self._fail_unless_reached(result, f"read {name}")
        if result.status_code == 404:
            return None
        if not result.ok:
            pytest.fail(f"Conjur refused to read {name}: HTTP {result.status_code} {result.body[:300]}")
        return result.body

    def destroy(self, name: str) -> None:
        self._update_root_policy("PATCH", f"- !delete\n  record: !variable {_policy_scalar(name)}\n", f"destroy {name}")


def conjur_from_env() -> Conjur:
    api_key: Final = os.environ.get(CYBERARK_API_KEY_ENV, "").strip()
    if not api_key:
        pytest.fail(f"The {SYSTEM} lane needs {CYBERARK_API_KEY_ENV} to reach its Conjur. {_START_HINT}")
    return Conjur(
        base_url=os.environ.get(CYBERARK_API_BASE_ENV, "").strip().rstrip("/") or DEFAULT_API_BASE,
        account=os.environ.get(CYBERARK_ACCOUNT_ENV, "").strip() or DEFAULT_ACCOUNT,
        username=os.environ.get(CYBERARK_USERNAME_ENV, "").strip() or DEFAULT_USERNAME,
        api_key=api_key,
    )


CYBERARK: Final = SecretBackend(system=SYSTEM, from_env=conjur_from_env, capabilities=frozenset())
