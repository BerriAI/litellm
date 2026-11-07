from typing import Final

from mcp.types import Tool

from litellm.mcp_client.output_schema_memory import OutputSchemaMemory


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _tool(name: str) -> Tool:
    return Tool(name=name, input_schema={"type": "object"}, output_schema={"type": "object", "title": name})


def _names(memory: OutputSchemaMemory, key: str) -> tuple[str, ...]:
    return tuple(sorted(tool.name for tool in memory.recall(key)))


def test_a_remembered_listing_is_recalled_until_it_expires() -> None:
    clock: Final = _Clock()
    memory: Final = OutputSchemaMemory(max_entries=4, clock=clock)
    memory.remember("upstream", [_tool("a")], ttl_seconds=60)

    clock.now = 59
    assert _names(memory, "upstream") == ("a",)
    clock.now = 60
    assert _names(memory, "upstream") == ()


def test_a_remembered_tool_keeps_its_output_schema() -> None:
    memory: Final = OutputSchemaMemory(max_entries=4)
    memory.remember("upstream", [_tool("a")], ttl_seconds=60)

    assert memory.recall("upstream")[0].output_schema == {"type": "object", "title": "a"}


def test_a_new_listing_replaces_the_remembered_one_and_restarts_its_expiry() -> None:
    clock: Final = _Clock()
    memory: Final = OutputSchemaMemory(max_entries=4, clock=clock)
    memory.remember("upstream", [_tool("a")], ttl_seconds=60)
    clock.now = 50
    memory.remember("upstream", [_tool("b")], ttl_seconds=60)
    assert _names(memory, "upstream") == ("b",)

    clock.now = 109
    assert _names(memory, "upstream") == ("b",)
    clock.now = 110
    assert _names(memory, "upstream") == ()


def test_the_oldest_upstream_is_dropped_when_the_memory_is_full() -> None:
    memory: Final = OutputSchemaMemory(max_entries=2)
    memory.remember("one", [_tool("a")], ttl_seconds=60)
    memory.remember("two", [_tool("a")], ttl_seconds=60)
    memory.remember("three", [_tool("a")], ttl_seconds=60)

    assert _names(memory, "one") == ()
    assert _names(memory, "two") == ("a",)
    assert _names(memory, "three") == ("a",)


def test_forgetting_an_upstream_drops_its_listing() -> None:
    memory: Final = OutputSchemaMemory(max_entries=4)
    memory.remember("upstream", [_tool("a")], ttl_seconds=60)
    memory.forget("upstream")

    assert _names(memory, "upstream") == ()
