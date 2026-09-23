import uuid
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway, eventually, string_value
from integration._support.database import read_rows
from integration._support.process import owned_proxy
from pydantic import JsonValue

WRITE_STATEMENT_MAX_BYTES: Final = 200_000
MESSAGES_PER_REQUEST: Final = 60
MESSAGE_CHARACTERS: Final = 2_000
REQUESTS: Final = 4


def _config_storing_prompts(tmp_path: Path) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["general_settings"]["store_prompts_in_spend_logs"] = True
    path: Final = tmp_path / "store-prompts.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _prompt_messages(marker: str) -> list[JsonValue]:
    return [
        {"role": "user", "content": f"{marker}-{index}-".ljust(MESSAGE_CHARACTERS, "x")}
        for index in range(MESSAGES_PER_REQUEST)
    ]


def _persisted(request_ids: tuple[str, ...]) -> list[dict[str, JsonValue]]:
    placeholders: Final = ", ".join("%s" for _ in request_ids)
    return read_rows(
        "SELECT request_id, xmin::text AS statement, octet_length(proxy_server_request::text) AS stored_bytes "
        f'FROM "LiteLLM_SpendLogs" WHERE request_id IN ({placeholders}) ORDER BY request_id',
        request_ids,
    )


@pytest.mark.covers("quota_management.spend_tracking.prompt_rows_are_written_in_byte_bounded_statements")
def test_prompt_carrying_spend_rows_flushed_together_are_written_in_byte_bounded_statements(
    gateway: Gateway, tmp_path: Path
) -> None:
    with (
        gateway.scenario() as scenario,
        owned_proxy(
            gateway,
            tmp_path,
            {
                "SPEND_LOG_WRITE_BATCH_MAX_BYTES": str(WRITE_STATEMENT_MAX_BYTES),
                "SPEND_LOG_QUEUE_POLL_INTERVAL": "15",
            },
            config=_config_storing_prompts(tmp_path),
        ) as owned,
    ):
        model: Final = scenario.model()
        request_ids: Final = tuple(
            string_value(
                owned.post(
                    "/v1/chat/completions",
                    {"model": model, "messages": _prompt_messages(f"integration-prompt-{uuid.uuid4().hex}")},
                )["id"]
            )
            for _ in range(REQUESTS)
        )
        assert len(set(request_ids)) == REQUESTS, request_ids
        rows: Final = eventually(lambda: _persisted(request_ids), lambda values: len(values) == REQUESTS, seconds=70)
    stored_bytes: Final = tuple(row["stored_bytes"] for row in rows)
    assert all(
        isinstance(size, int) and WRITE_STATEMENT_MAX_BYTES // 2 < size < WRITE_STATEMENT_MAX_BYTES
        for size in stored_bytes
    ), rows
    assert len({row["statement"] for row in rows}) == REQUESTS, rows
