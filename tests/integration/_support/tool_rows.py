import json
import sys
from typing import Final, LiteralString

from integration._support.database import write_rows

CLEAR_QUERY: Final[LiteralString] = 'DELETE FROM "LiteLLM_ToolTable" WHERE tool_name = %s'


def clear(tool_names: tuple[str, ...]) -> None:
    for tool_name in tool_names:
        write_rows(CLEAR_QUERY, (tool_name,))


if __name__ == "__main__":
    if sys.argv[1] != "clear":
        raise SystemExit(f"unknown command: {sys.argv[1]}")
    clear(tuple(sys.argv[2:]))
    sys.stdout.write(json.dumps({"cleared": sys.argv[2:]}) + "\n")
