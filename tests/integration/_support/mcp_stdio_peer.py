import sys
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[2])

import asyncio  # noqa: E402  # the script directory holds mcp.py, which would shadow the mcp package
import json  # noqa: E402
import os  # noqa: E402
from typing import Final  # noqa: E402

import anyio  # noqa: E402
from integration._support.mcp import math_service  # noqa: E402
from mcp.server.stdio import stdio_server  # noqa: E402


class Recording:
    def __init__(self, source: anyio.AsyncFile[str], record: Path) -> None:
        self.source = source
        self.record = record

    def __aiter__(self) -> "Recording":
        return self

    async def __anext__(self) -> str:
        line: Final = await self.source.readline()
        if not line:
            raise StopAsyncIteration
        with self.record.open("a") as sink:
            passed: Final = {name: value for name, value in os.environ.items() if name.startswith("PEER_")}
            sink.write(json.dumps({"body": json.loads(line), "env": passed}) + "\n")
        return line

    async def readline(self) -> str:
        return await self.__anext__()


async def main() -> None:
    record: Final = Path(sys.argv[1])
    service: Final = math_service("integration-stdio", rich=sys.argv[2] == "rich")
    stdin: Final = anyio.wrap_file(sys.stdin)
    async with stdio_server(stdin=Recording(stdin, record)) as (read_stream, write_stream):
        lowlevel: Final = service._lowlevel_server
        await lowlevel.run(read_stream, write_stream, lowlevel.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
