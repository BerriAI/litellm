import uuid
from typing import Final

import httpx

from tests.integration._support.client import Gateway, list_value, object_value


def test_a_non_admin_key_cannot_generate_keys_even_after_an_admin_ui_login(gateway: Gateway) -> None:
    user_id: Final = f"integration-key-owner-{uuid.uuid4().hex}"
    with gateway.scenario() as scenario, httpx.Client(
        base_url=str(gateway.client.base_url), timeout=15, trust_env=False
    ) as browser:
        user_key: Final = scenario.key(user_id=user_id)
        login: Final = browser.post("/login", data={"username": "admin", "password": gateway.key})
        assert login.status_code in (200, 303), login.text
        assert browser.cookies.get("token") is not None

        denied: Final = browser.post(
            "/key/generate",
            json={"user_id": user_id},
            headers={"Authorization": f"Bearer {user_key}"},
        )
        owned_keys: Final = list_value(gateway.get("/key/list", {"user_id": user_id})["keys"])

    assert denied.status_code == 401, denied.text
    error: Final = object_value(denied.json()["error"])
    assert error["type"] == "auth_error"
    assert "Only proxy admin can be used to generate" in str(error["message"])
    assert len(owned_keys) == 1
