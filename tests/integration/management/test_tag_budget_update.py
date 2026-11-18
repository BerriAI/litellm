from typing import Final
from uuid import uuid4

from tests.integration._support.client import Gateway, object_value, string_value


def test_tag_budget_can_be_updated_with_its_existing_budget_id(gateway: Gateway) -> None:
    name: Final = f"integration-tag-{uuid4().hex}"
    gateway.post("/tag/new", {"name": name, "max_budget": 10})
    with gateway.scenario() as scenario:
        scenario.cleanups.callback(gateway.post, "/tag/delete", {"name": name})
        before: Final = object_value(gateway.post("/tag/info", {"names": [name]})[name])
        budget_id: Final = string_value(object_value(before["litellm_budget_table"])["budget_id"])
        scenario.cleanups.callback(gateway.post, "/budget/delete", {"id": budget_id})
        response: Final = gateway.request(
            "POST", "/tag/update", {"name": name, "budget_id": budget_id, "max_budget": 20}
        )
        assert response.status_code == 200, response.text
        after: Final = object_value(gateway.post("/tag/info", {"names": [name]})[name])
        assert object_value(after["litellm_budget_table"])["max_budget"] == 20, after
