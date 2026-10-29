import uuid
from pathlib import Path
from typing import Final

from integration._support.client import Gateway, object_value
from integration._support.database import scratch_database
from integration._support.process import owned_proxy

MASTER_KEY_USER: Final = "default_user_id"


def test_master_key_invites_on_a_proxy_where_nobody_signed_in(gateway: Gateway, tmp_path: Path) -> None:
    with (
        scratch_database() as database_url,
        owned_proxy(
            gateway, tmp_path, {"DATABASE_URL": database_url}, remove_environment=("DATABASE_URL_READ_REPLICA",)
        ) as proxy,
        proxy.scenario() as scenario,
    ):
        invitee: Final = scenario.user(user_role="proxy_admin")
        created: Final = proxy.request("POST", "/invitation/new", {"user_id": invitee})
        assert created.status_code == 200, created.text
        invitation: Final = object_value(created.json())
        assert (invitation["user_id"], invitation["created_by"], invitation["updated_by"]) == (
            invitee,
            MASTER_KEY_USER,
            MASTER_KEY_USER,
        )
        accepted: Final = proxy.request(
            "POST", "/invitation/update", {"invitation_id": invitation["id"], "is_accepted": True}
        )
        assert accepted.status_code == 200, accepted.text
        assert object_value(accepted.json())["is_accepted"] is True
        assert proxy.rows('SELECT user_id FROM "LiteLLM_UserTable" WHERE user_id = %s', (MASTER_KEY_USER,)) == []
        missing: Final = proxy.request("POST", "/invitation/new", {"user_id": f"integration-{uuid.uuid4().hex}"})
        assert missing.status_code == 400, missing.text
        assert "User id does not exist in 'LiteLLM_UserTable'" in missing.text
