import json
import uuid
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from typing import Final

import pytest

from integration._support.client import Gateway, delete_key_if_present, eventually
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server


@pytest.mark.covers("quota_management.spend_tracking.filtered_ledger_preserves_owner_identity_and_totals")
def test_rotated_keys_users_and_model_groups_preserve_success_failure_cache_ledger(gateway: Gateway) -> None:
    def provider(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/chat/completions"
        body: Final = json.loads(request.body)
        if body["messages"][-1]["content"].endswith("reject"):
            return Reply(status=400, body=b'{"error":{"message":"synthetic ledger rejection","type":"invalid_request_error","code":"400"}}')
        return Reply(body=json.dumps({"id": "chatcmpl-" + uuid.uuid4().hex, "object": "chat.completion", "created": 1, "model": "gpt-4o-mini", "choices": [{"index": 0, "message": {"role": "assistant", "content": "synthetic ledger answer"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 20, "completion_tokens": 20, "total_tokens": 40}}).encode())

    with wire_server(provider) as wire, gateway.scenario() as scenario:
        owners: Final = (scenario.user(), scenario.user())
        models: Final = tuple(scenario.model(api_base=wire.url + "/v1", input_cost_per_token=0.001, output_cost_per_token=0.002, num_retries=0) for _ in owners)
        keys = []
        for owner, model in zip(owners, models, strict=True):
            created: Final = gateway.post("/key/generate", {"user_id": owner, "models": [model]})["key"]
            scenario.cleanups.callback(delete_key_if_present, gateway, created)
            keys.append(created)
        rotated: Final = "sk-" + uuid.uuid4().hex
        scenario.cleanups.callback(delete_key_if_present, gateway, rotated)
        changed: Final = gateway.post("/key/regenerate", {"key": keys[0], "new_key": rotated, "grace_period": "0s"})
        assert changed["key"] == rotated
        active: Final = (rotated, keys[1])
        digests: Final = tuple(sha256(key.encode()).hexdigest() for key in active)
        ledger: dict[str, tuple[str, str, str]] = {}
        for owner, model, key, digest in zip(owners, models, active, digests, strict=True):
            assert read_rows('SELECT user_id FROM "LiteLLM_VerificationToken" WHERE token=%s', (digest,)) == [{"user_id": owner}]
            prompt: Final = uuid.uuid4().hex
            replies = []
            for index in range(2):
                result: Final = gateway.request("POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": prompt}], "metadata": {"integration_marker": f"{prompt}-{index}"}}, key=key)
                assert result.status_code == 200, result.text
                body: Final = result.json()
                assert body["choices"][0]["message"]["content"] == "synthetic ledger answer"
                assert body["usage"]["prompt_tokens"] == 20 and body["usage"]["completion_tokens"] == 20 and body["usage"]["total_tokens"] == 40
                replies.append(body["id"])
            assert replies[0] == replies[1]
            rejected: Final = gateway.request("POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": prompt + "reject"}]}, key=key)
            assert rejected.status_code == 400 and "synthetic ledger rejection" in rejected.text
            ledger[digest] = (replies[0], rejected.headers["x-litellm-call-id"], model)
        observed: Final = wire.drain()
        assert len(observed) == 4
        assert sum(json.loads(item.body)["messages"][-1]["content"].endswith("reject") for item in observed) == 2
        rows: Final = eventually(lambda: read_rows('SELECT request_id, api_key, "user", model_group, status, cache_hit, spend, prompt_tokens, completion_tokens FROM "LiteLLM_SpendLogs" WHERE api_key=ANY(%s)', (list(digests),)), lambda values: len(values) == 6, seconds=70)
        assert len({row["request_id"] for row in rows}) == 6
        assert sum(float(row["spend"]) for row in rows) == pytest.approx(0.12)
        now: Final = datetime.now(timezone.utc)
        window: Final = {"start_date": (now - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"), "end_date": (now + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"), "page_size": "100"}
        for owner, key, digest in zip(owners, active, digests, strict=True):
            identity, failure, model = ledger[digest]
            selected: Final = tuple(row for row in rows if row["api_key"] == digest)
            assert len(selected) == 3 and all(row["user"] == owner and row["model_group"] == model for row in selected)
            assert sum(row["status"] == "success" for row in selected) == 2
            assert sum(row["status"] == "failure" for row in selected) == 1
            assert sum(str(row["cache_hit"]).lower() == "true" for row in selected) == 1
            assert sorted(float(row["spend"]) for row in selected) == [0, 0, 0.06]
            for row in selected:
                hit: Final = str(row["cache_hit"]).lower() == "true"
                if row["request_id"] == identity:
                    assert row["status"] == "success" and not hit and float(row["spend"]) == pytest.approx(0.06)
                elif row["request_id"] == failure:
                    assert row["status"] == "failure" and not hit and float(row["spend"]) == 0
                    assert row["completion_tokens"] == 0
                else:
                    assert row["request_id"].startswith(identity + "_cache_hit")
                    assert row["status"] == "success" and hit and float(row["spend"]) == 0
                if row["status"] == "success":
                    assert row["prompt_tokens"] == 20 and row["completion_tokens"] == 20
            expected: Final = {row["request_id"] for row in selected}
            def projection(row):
                return (row["request_id"], row["api_key"], row["user"], row["model_group"], row["status"], str(row["cache_hit"]).lower(), float(row["spend"]), row["prompt_tokens"], row["completion_tokens"])
            projected: Final = sorted(projection(row) for row in selected)
            for query in ({"api_key": digest}, {"user_id": owner}, {"model_group": model}):
                filtered: Final = gateway.get("/spend/logs/v2", params={**window, **query})
                assert filtered["total"] == 3 and filtered["total_is_capped"] is False
                assert len(filtered["data"]) == 3
                assert {row["request_id"] for row in filtered["data"]} == expected
                assert sorted(projection(row) for row in filtered["data"]) == projected
            for token in (key, digest):
                legacy: Final = gateway.request("GET", "/spend/logs", params={"api_key": token})
                assert legacy.status_code == 200, legacy.text
                assert len(legacy.json()) == 3
                assert {row["request_id"] for row in legacy.json()} == expected
                assert sorted(projection(row) for row in legacy.json()) == projected
