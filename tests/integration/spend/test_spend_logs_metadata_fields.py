import json
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from typing import Final
from uuid import uuid4

from integration._support.client import Gateway, eventually, object_value, string_value
from integration._support.database import read_rows
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

CACHED_PROMPT_TOKENS: Final = 4


def _config(tmp_path: Path, spend_logs_metadata_fields: Mapping[str, JsonValue]) -> Path:
    config: Final = tmp_path / f"spend-logs-metadata-fields-{uuid4()}.json"
    config.write_text(
        json.dumps(
            {
                "model_list": [],
                "general_settings": {
                    "master_key": "os.environ/LITELLM_MASTER_KEY",
                    "database_url": "os.environ/DATABASE_URL",
                    "store_model_in_db": True,
                    "proxy_batch_write_at": 1,
                    "proxy_batch_polling_interval": 1,
                    "spend_logs_metadata_fields": dict(spend_logs_metadata_fields),
                },
            }
        )
    )
    return config


def _respond(request: Request) -> Reply:
    if request.method == "GET":
        return Reply(body=b'{"object":"list","data":[]}')
    return Reply(
        body=json.dumps(
            {
                "id": f"chatcmpl-{uuid4()}",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                    "prompt_tokens_details": {"cached_tokens": CACHED_PROMPT_TOKENS},
                },
            }
        ).encode()
    )


def _stored_row_after_one_chat(
    gateway: Gateway, tmp_path: Path, spend_logs_metadata_fields: Mapping[str, JsonValue]
) -> tuple[dict[str, JsonValue], str]:
    with (
        wire_server(_respond) as wire,
        owned_proxy(gateway, tmp_path, {}, config=_config(tmp_path, spend_logs_metadata_fields)) as isolated,
        isolated.scenario() as scenario,
    ):
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=wire.url, api_key="synthetic-openai-key")
        api_key: Final = scenario.key(key_alias=f"metadata-fields-{uuid4()}", models=[model])
        response_id: Final = string_value(isolated.chat(model, key=api_key)["id"])
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT metadata, proxy_server_request, response FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
                (response_id,),
            ),
            lambda values: len(values) == 1,
            seconds=70,
        )
        return rows[0], sha256(api_key.encode()).hexdigest()


def test_excluded_metadata_fields_are_not_stored_but_still_reach_daily_spend(gateway: Gateway, tmp_path: Path) -> None:
    row, hashed_key = _stored_row_after_one_chat(
        gateway, tmp_path, {"exclude": ["model_map_information", "usage_object"]}
    )

    metadata: Final = object_value(row["metadata"])
    assert "model_map_information" not in metadata, metadata
    assert "usage_object" not in metadata, metadata
    assert {"status", "cold_storage_object_key"} <= set(metadata), metadata
    assert string_value(metadata["user_api_key_alias"]).startswith("metadata-fields-")
    assert row["proxy_server_request"] == {}
    assert row["response"] == {}
    daily: Final = eventually(
        lambda: read_rows(
            'SELECT prompt_tokens, cache_read_input_tokens FROM "LiteLLM_DailyUserSpend" WHERE api_key=%s',
            (hashed_key,),
        ),
        lambda values: len(values) == 1,
        seconds=70,
    )
    assert daily[0]["cache_read_input_tokens"] == CACHED_PROMPT_TOKENS, daily


def test_included_metadata_fields_are_the_only_ones_stored_besides_always_kept(
    gateway: Gateway, tmp_path: Path
) -> None:
    row, _ = _stored_row_after_one_chat(gateway, tmp_path, {"include": ["user_api_key_alias"]})

    assert set(object_value(row["metadata"])) == {"status", "cold_storage_object_key", "user_api_key_alias"}
