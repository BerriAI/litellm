from typing import Final

from tests.integration._support.client import Gateway


def test_user_key_cannot_generate_another_key(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        user: Final = scenario.user(user_role="internal_user")
        user_key: Final = scenario.key(user_id=user)

        response: Final = gateway.request(
            "POST",
            "/key/generate",
            {"user_id": user},
            key=user_key,
        )

    assert response.status_code == 403, response.text
    assert response.json()["error"]["type"] == "auth_error"
