from typing import Final
from hashlib import sha256

import pytest

from tests.integration._support.client import Gateway, object_value
from tests.integration._support.database import read_rows


@pytest.mark.covers("mgmt.key.update.preserves_independent_fields")
def test_update_preserves_independent_fields_and_serving(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model], key_alias="before", metadata={"retained": "value"})
        gateway.chat(model, key=key)
        gateway.post("/key/update", {"key": key, "key_alias": "after"})
        info: Final = object_value(gateway.get("/key/info", {"key": key})["info"])
        assert info["key_alias"] == "after"
        assert info["models"] == [model]
        assert object_value(info["metadata"])["retained"] == "value"
        response: Final = gateway.chat(model, key=key)
        assert object_value(response["usage"])["total_tokens"] == 40
        replacement: Final = scenario.model()
        gateway.post("/key/update", {"key": key, "models": [replacement]})
        saved: Final = read_rows(
            'SELECT key_alias, models, metadata FROM "LiteLLM_VerificationToken" WHERE token = %s',
            (sha256(key.encode()).hexdigest(),),
        )
        assert len(saved) == 1
        assert saved[0]["models"] == [replacement]
        assert saved[0]["key_alias"] == "after"
        denied: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": "old grant"}]},
            key=key,
        )
        assert denied.status_code == 403, denied.text
        assert object_value(object_value(denied.json())["error"])["type"] == "key_model_access_denied"
        assert object_value(gateway.chat(replacement, key=key)["usage"])["total_tokens"] == 40
