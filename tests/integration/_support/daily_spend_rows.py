import json
import sys
from datetime import datetime, timezone
from typing import Final, LiteralString

from integration._support.database import read_rows, write_rows

SEED_QUERY: Final[LiteralString] = """
INSERT INTO "LiteLLM_DailyUserSpend" (
    id, user_id, date, api_key, model, model_group, custom_llm_provider,
    endpoint, mcp_namespaced_tool_name,
    prompt_tokens, completion_tokens, spend,
    api_requests, successful_requests, failed_requests, updated_at
) VALUES (
    gen_random_uuid()::text, %s, %s, %s, %s, %s, 'bedrock',
    NULL, NULL,
    %s, %s, %s,
    %s, %s, %s, now()
)
"""

CLEAR_COUNT_QUERY: Final[LiteralString] = 'SELECT count(*) AS count FROM "LiteLLM_DailyUserSpend" WHERE api_key=%s'
CLEAR_QUERY: Final[LiteralString] = 'DELETE FROM "LiteLLM_DailyUserSpend" WHERE api_key=%s'


def seed(api_key: str, group: str, deployment: str) -> int:
    today: Final = datetime.now(timezone.utc).date().isoformat()
    write_rows(
        SEED_QUERY,
        (api_key, today, api_key, deployment, group, "270000", "1000", "0.81", "270", "270", "0"),
    )
    write_rows(
        SEED_QUERY,
        (api_key, today, api_key, group, "", "5000", "0", "0.0", "50", "0", "50"),
    )
    return 2


def clear(api_key: str) -> int:
    count: Final = int(str(read_rows(CLEAR_COUNT_QUERY, (api_key,))[0]["count"]))
    write_rows(CLEAR_QUERY, (api_key,))
    return count


if __name__ == "__main__":
    command: Final = sys.argv[1]
    if command == "seed":
        sys.stdout.write(json.dumps({"affected": seed(sys.argv[2], sys.argv[3], sys.argv[4])}) + "\n")
    elif command == "clear":
        sys.stdout.write(json.dumps({"affected": clear(sys.argv[2])}) + "\n")
    else:
        raise SystemExit(f"unknown command: {command}")
