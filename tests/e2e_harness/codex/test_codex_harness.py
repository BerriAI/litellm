from __future__ import annotations

import time
import tomllib
from pathlib import Path
from typing import Final

from cli_harness.rate_limiter import RateBucket, RateLimiter
from codex.harness import McpServer, codex_harness


def test_an_added_mcp_server_is_a_table_codex_reads_next_to_the_gateway_provider(tmp_path: Path) -> None:
    server: Final = McpServer(
        command='/opt/py "3.14"/bin/python',
        args=("C:\\tools\\mcp_server.py", "--flag"),
        env={"SECRET_FILE": "/tmp/a dir/secret.txt", "OTHER": "x"},
    )
    with codex_harness(
        gateway="http://127.0.0.1:4000",
        api_key="unused",
        rate_bucket=RateBucket.OPENAI,
        limiter=RateLimiter({}, tmp_path, time.monotonic, time.sleep),
    ) as codex:
        codex.add_mcp_server("vault", server)
        config: Final = tomllib.loads((codex.codex_home / "config.toml").read_text())

    assert config["model_provider"] == "litellm"
    assert config["mcp_servers"] == {
        "vault": {
            "command": server.command,
            "args": list(server.args),
            "env": dict(server.env),
            "default_tools_approval_mode": "approve",
        }
    }
