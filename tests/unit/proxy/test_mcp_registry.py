import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Final

import litellm.proxy

REGISTRY_SERVERS: Final[Sequence[Mapping[str, object]]] = json.loads(
    (Path(litellm.proxy.__file__).parent / "mcp_registry.json").read_text()
)["servers"]


def _scoped_npm_packages() -> Iterator[str]:
    for server in REGISTRY_SERVERS:
        if server.get("command") != "npx":
            continue
        args = server.get("args")
        yield from (arg for arg in args if isinstance(arg, str) and arg.startswith("@")) if isinstance(args, list) else ()


def test_no_stdio_server_installs_from_the_nonexistent_anthropic_npm_scope() -> None:
    """
    https://github.com/BerriAI/litellm/issues/24769 - picking one of these from the discovery
    page failed tool discovery with an npm 404. registry.npmjs.org answered 404 for every
    `@anthropic/mcp-server-*` name the registry used when checked on 2026-09-29
    """
    packages: Final = tuple(_scoped_npm_packages())

    assert packages
    assert [package for package in packages if package.startswith("@anthropic/")] == []
