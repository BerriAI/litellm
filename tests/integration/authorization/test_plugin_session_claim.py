import base64
import hashlib
import hmac
import os
import signal
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final

import psutil
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from integration._support.client import Gateway, eventually, string_value
from integration._support.process import owned_proxy, owned_proxy_process
from pydantic import JsonValue, TypeAdapter

_SALT: Final = "sk-integration-plugin-salt"
_PLUGIN: Final = "integration-plugin"
_OTHER_PLUGIN: Final = "integration-other-plugin"
_GCM_PREFIX: Final = "v2:gcm:"
_TTL_SECONDS: Final = 30
_CLAIM_ADAPTER: Final = TypeAdapter(dict[str, JsonValue])


def _plugin_config(directory: Path) -> Path:
    config: Final = directory / "proxy_config.yaml"
    config.write_text(
        "model_list: []\n"
        "general_settings:\n"
        "  master_key: os.environ/LITELLM_MASTER_KEY\n"
        "  database_url: os.environ/DATABASE_URL\n"
        "  store_model_in_db: true\n"
        "  plugins:\n"
        f"    - name: {_PLUGIN}\n"
        "      url: http://127.0.0.1:9\n"
        f"    - name: {_OTHER_PLUGIN}\n"
        "      url: http://127.0.0.1:9\n"
    )
    return config


def _plugin_key(plugin_name: str, salt: str = _SALT) -> bytes:
    return hmac.new(salt.encode(), plugin_name.encode(), hashlib.sha256).digest()


def _decrypt_claim(claim: str, key: bytes) -> dict[str, JsonValue]:
    assert claim.startswith(_GCM_PREFIX), claim
    raw: Final = base64.urlsafe_b64decode(claim[len(_GCM_PREFIX) :])
    return _CLAIM_ADAPTER.validate_json(AESGCM(key).decrypt(raw[:12], raw[12:], None))


def _issue(candidate: Gateway, key: str, plugin_name: str = _PLUGIN) -> str:
    response: Final = candidate.request("GET", "/api/plugins/auth-token", key=key, params={"plugin_name": plugin_name})
    assert response.status_code == 200, response.text
    return string_value(_CLAIM_ADAPTER.validate_json(response.text)["session_claim"])


def test_plugin_claim_is_aes_gcm_under_the_hmac_derived_plugin_key(gateway: Gateway, tmp_path: Path) -> None:
    with owned_proxy(gateway, tmp_path, {"LITELLM_SALT_KEY": _SALT}, config=_plugin_config(tmp_path)) as candidate:
        with candidate.scenario() as scenario:
            user_id: Final = scenario.user(user_id=f"plugin-admin-{uuid.uuid4().hex}", user_role="proxy_admin")
            key: Final = scenario.key(user_id=user_id)
            issued_at: Final = int(time.time())
            claim: Final = _issue(candidate, key)
            decrypted: Final = _decrypt_claim(claim, _plugin_key(_PLUGIN))
            assert decrypted["plugin"] == _PLUGIN, decrypted
            assert decrypted["user_id"] == user_id, decrypted
            assert isinstance(decrypted["user_role"], str), decrypted
            expiry: Final = decrypted["exp"]
            assert isinstance(expiry, int), decrypted
            assert issued_at + _TTL_SECONDS <= expiry <= issued_at + _TTL_SECONDS + 5, decrypted


def test_plugin_claim_rejects_other_plugin_key_wrong_salt_and_tampering(gateway: Gateway, tmp_path: Path) -> None:
    with owned_proxy(gateway, tmp_path, {"LITELLM_SALT_KEY": _SALT}, config=_plugin_config(tmp_path)) as candidate:
        claim: Final = _issue(candidate, candidate.key)
        raw: Final = base64.urlsafe_b64decode(claim[len(_GCM_PREFIX) :])
        flipped: Final = raw[:-1] + bytes([raw[-1] ^ 0x01])
        tampered: Final = _GCM_PREFIX + base64.urlsafe_b64encode(flipped).decode()
        for wrong_key in (_plugin_key(_OTHER_PLUGIN), _plugin_key(_PLUGIN, salt="sk-not-the-proxy-salt")):
            try:
                _decrypt_claim(claim, wrong_key)
            except InvalidTag:
                continue
            raise AssertionError(f"claim for {_PLUGIN} decrypted under a foreign key: {claim}")
        try:
            _decrypt_claim(tampered, _plugin_key(_PLUGIN))
        except InvalidTag:
            pass
        else:
            raise AssertionError(f"tampered claim decrypted: {tampered}")
        assert _decrypt_claim(claim, _plugin_key(_PLUGIN))["plugin"] == _PLUGIN


def test_plugin_auth_token_still_gates_registration_and_bearer(gateway: Gateway, tmp_path: Path) -> None:
    with owned_proxy(gateway, tmp_path, {"LITELLM_SALT_KEY": _SALT}, config=_plugin_config(tmp_path)) as candidate:
        missing: Final = candidate.request("GET", "/api/plugins/auth-token", params={"plugin_name": "not-registered"})
        assert missing.status_code == 404, missing.text
        assert missing.json()["detail"] == "Plugin 'not-registered' is not registered."
        anonymous: Final = candidate.client.get("/api/plugins/auth-token", params={"plugin_name": _PLUGIN})
        assert anonymous.status_code == 401, anonymous.text
        assert _decrypt_claim(_issue(candidate, candidate.key), _plugin_key(_PLUGIN))["plugin"] == _PLUGIN


def test_plugin_claims_survive_a_worker_kill_during_a_burst(gateway: Gateway, tmp_path: Path) -> None:
    with owned_proxy_process(
        gateway, tmp_path, {"LITELLM_SALT_KEY": _SALT}, config=_plugin_config(tmp_path), workers=2
    ) as owned:
        candidate: Final = owned.gateway
        root: Final = psutil.Process(owned.process.pid)
        workers: Final = eventually(
            lambda: tuple(root.children(recursive=True)), lambda found: len(found) >= 2, seconds=30
        )
        with ThreadPoolExecutor(max_workers=8) as pool:
            before: Final = tuple(
                future.result() for future in [pool.submit(_issue, candidate, candidate.key) for _ in range(12)]
            )
            os.kill(workers[0].pid, signal.SIGKILL)
            probe: Final = eventually(
                lambda: candidate.request("GET", "/api/plugins/auth-token", params={"plugin_name": _PLUGIN}),
                lambda response: response.status_code == 200,
                seconds=30,
            )
            after: Final = tuple(
                future.result() for future in [pool.submit(_issue, candidate, candidate.key) for _ in range(12)]
            )
        claims: Final = before + (string_value(_CLAIM_ADAPTER.validate_json(probe.text)["session_claim"]),) + after
        assert len(set(claims)) == len(claims), "nonce reuse: two claims shared ciphertext"
        decrypted: Final = tuple(_decrypt_claim(claim, _plugin_key(_PLUGIN)) for claim in claims)
        assert all(value["plugin"] == _PLUGIN for value in decrypted), decrypted
