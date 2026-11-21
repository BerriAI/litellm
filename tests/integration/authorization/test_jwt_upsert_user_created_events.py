import json
import time
import uuid
from pathlib import Path
from queue import Empty
from typing import Final

import jwt
import pytest
import yaml
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm
from pydantic import JsonValue

from tests.integration._support.client import Gateway, eventually
from tests.integration._support.database import read_rows
from tests.integration._support.process import owned_proxy
from tests.integration._support.wire import Reply, Request, Wire, wire_server

pytestmark: Final = pytest.mark.shares_worker_state

KEY_ID: Final = "integration-jwt-user-created-key"


def _drain_alerts(sink: Wire) -> tuple[str, ...]:
    def _next() -> Request | None:
        try:
            return sink.received.get_nowait()
        except Empty:
            return None

    requests: Final = tuple(iter(_next, None))
    return tuple(json.loads(request.body)["text"] for request in requests if request.method == "POST")


def _created_audit_rows(user_id: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT object_id, action, table_name, changed_by FROM "LiteLLM_AuditLog" WHERE object_id = %s AND action = %s',
        (user_id, "created"),
    )


def test_jwt_upserted_user_fires_the_user_created_alert_and_audit_log_once(gateway: Gateway, tmp_path: Path) -> None:
    private_key: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_jwk: Final = RSAAlgorithm.to_jwk(private_key.public_key())
    jwks_body: Final = json.dumps({"keys": [{**json.loads(public_jwk), "kid": KEY_ID}]}).encode()

    with (
        wire_server(lambda _: Reply(body=jwks_body)) as jwks,
        wire_server(lambda _: Reply(body=b"ok", content_type="text/plain")) as alerts,
        gateway.scenario() as scenario,
    ):
        config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        config["general_settings"] = {
            **config["general_settings"],
            "enable_jwt_auth": True,
            "litellm_jwtauth": {"user_id_jwt_field": "sub", "user_id_upsert": True},
            "alerting": ["slack"],
            "alert_types": ["new_internal_user_created"],
            "alert_to_webhook_url": {"new_internal_user_created": alerts.url},
        }
        config["litellm_settings"] = {**config["litellm_settings"], "store_audit_logs": True}
        path: Final = tmp_path / "jwt_user_created_events.yaml"
        path.write_text(yaml.safe_dump(config))
        manual_user: Final = f"integration-manual-{uuid.uuid4().hex}"
        subject: Final = f"integration-jwt-{uuid.uuid4().hex}"
        token: Final = jwt.encode(
            {"sub": subject, "iat": int(time.time()), "exp": int(time.time()) + 300},
            private_key,
            algorithm="RS256",
            headers={"kid": KEY_ID},
        )
        with owned_proxy(gateway, tmp_path, {"JWT_PUBLIC_KEY_URL": jwks.url}, config=path) as candidate:
            model: Final = scenario.model()
            scenario.cleanups.callback(scenario.delete_user, manual_user)

            created: Final = candidate.request("POST", "/user/new", {"user_id": manual_user, "auto_create_key": False})
            assert created.status_code == 200, created.text
            manual_alerts: Final = eventually(lambda: _drain_alerts(alerts), lambda found: len(found) > 0, seconds=30)
            assert len(manual_alerts) == 1 and manual_user in manual_alerts[0], manual_alerts
            assert len(eventually(lambda: _created_audit_rows(manual_user), lambda rows: len(rows) == 1)) == 1

            def complete() -> None:
                response = candidate.request(
                    "POST",
                    "/v1/chat/completions",
                    {"model": model, "messages": [{"role": "user", "content": "user created event control"}]},
                    key=token,
                )
                assert response.status_code == 200, response.text

            complete()
            scenario.cleanups.callback(scenario.delete_user, subject)
            assert read_rows('SELECT user_id FROM "LiteLLM_UserTable" WHERE user_id = %s', (subject,)) == [
                {"user_id": subject}
            ]
            jwt_alerts: Final = eventually(
                lambda: _drain_alerts(alerts), lambda found: len(found) > 0, seconds=30, return_last_on_timeout=True
            )
            assert len(jwt_alerts) == 1, f"expected one New Internal User Created alert for {subject}, got {jwt_alerts}"
            assert "New Internal User Created" in jwt_alerts[0] and subject in jwt_alerts[0], jwt_alerts
            audit_rows: Final = eventually(
                lambda: _created_audit_rows(subject), lambda rows: len(rows) == 1, return_last_on_timeout=True
            )
            assert audit_rows == [
                {"object_id": subject, "action": "created", "table_name": "LiteLLM_UserTable", "changed_by": subject}
            ]

            complete()
            assert _drain_alerts(alerts) == (), "a returning JWT user is not announced as created again"
            assert len(_created_audit_rows(subject)) == 1
