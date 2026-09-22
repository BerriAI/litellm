import uuid
from typing import Final

import pytest

from tests.integration._support.client import Gateway, object_value

CONFIG_STORE_ID: Final = "vs_integration_config_store"


def listed_store(gateway: Gateway, vector_store_id: str) -> dict[str, object]:
    listed: Final = gateway.get("/vector_store/list")
    rows: Final = listed["data"]
    assert isinstance(rows, list)
    matches: Final = tuple(object_value(row) for row in rows if object_value(row)["vector_store_id"] == vector_store_id)
    assert len(matches) == 1, f"{vector_store_id} appears {len(matches)} times in {listed}"
    return matches[0]


@pytest.mark.covers("mgmt.vector_store.list.keeps_config_store_beside_db_stores")
def test_config_store_is_listed_beside_db_store_and_survives_listing(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        db_store_id: Final = f"vs_db_{uuid.uuid4().hex}"
        gateway.post("/vector_store/new", {"vector_store_id": db_store_id, "custom_llm_provider": "openai"})
        scenario.cleanups.callback(gateway.post, "/vector_store/delete", {"vector_store_id": db_store_id})
        before: Final = object_value(
            gateway.post("/vector_store/info", {"vector_store_id": CONFIG_STORE_ID})["vector_store"]
        )
        assert before["vector_store_id"] == CONFIG_STORE_ID, before

        config_row: Final = listed_store(gateway, CONFIG_STORE_ID)
        assert config_row["is_config"] is True, config_row
        assert config_row["vector_store_name"] == "integration-config-store", config_row
        db_row: Final = listed_store(gateway, db_store_id)
        assert db_row["is_config"] is False, db_row

        after: Final = object_value(
            gateway.post("/vector_store/info", {"vector_store_id": CONFIG_STORE_ID})["vector_store"]
        )
        assert after["vector_store_id"] == CONFIG_STORE_ID, after
        assert after["is_config"] is True, after
        assert listed_store(gateway, CONFIG_STORE_ID)["is_config"] is True


@pytest.mark.covers("mgmt.vector_store.write.config_store_is_read_only")
def test_config_store_refuses_new_update_and_delete(gateway: Gateway) -> None:
    for path, body in (
        ("/vector_store/update", {"vector_store_id": CONFIG_STORE_ID, "vector_store_name": "renamed"}),
        ("/vector_store/delete", {"vector_store_id": CONFIG_STORE_ID}),
        ("/vector_store/new", {"vector_store_id": CONFIG_STORE_ID, "custom_llm_provider": "openai"}),
    ):
        refused: Final = gateway.request("POST", path, body)
        assert refused.status_code == 400, f"{path}: {refused.status_code} {refused.text}"
        error: Final = object_value(object_value(refused.json())["detail"])
        assert error["vector_store_id"] == CONFIG_STORE_ID, refused.text
        assert "config file" in str(error["error"]), refused.text
    row: Final = listed_store(gateway, CONFIG_STORE_ID)
    assert row["vector_store_name"] == "integration-config-store", row
    assert row["is_config"] is True, row
    info: Final = object_value(gateway.post("/vector_store/info", {"vector_store_id": CONFIG_STORE_ID})["vector_store"])
    assert info["vector_store_name"] == "integration-config-store", info
