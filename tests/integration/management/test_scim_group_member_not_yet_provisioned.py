import uuid
from typing import Final

from integration._support.client import Gateway, object_value, string_value
from pydantic import JsonValue


def test_scim_group_patch_add_member_provisions_the_missing_user(gateway: Gateway) -> None:
    missing_user: Final = f"scim-pending-{uuid.uuid4().hex}"

    with gateway.scenario() as scenario:
        team: Final = scenario.team()
        response: Final = gateway.request(
            "PATCH",
            f"/scim/v2/Groups/{team}",
            {
                "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
                "Operations": [
                    {"op": "add", "path": "members", "value": [{"value": missing_user}]}
                ],
            },
        )
        scenario.cleanups.callback(scenario.delete_user, missing_user)
        assert response.status_code == 200, response.text
        team_info: dict[str, JsonValue] = gateway.get("/team/info", {"team_id": team})
        members: Final = object_value(team_info["team_info"]).get("members_with_roles") or []
        member_ids: Final = [
            string_value(object_value(member)["user_id"]) for member in members if isinstance(member, dict)
        ]
        assert missing_user in member_ids, members
