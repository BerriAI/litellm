from types import SimpleNamespace

from litellm.proxy._experimental.mcp_server.utils import (
    get_possible_server_name_prefixes,
    split_server_prefix_from_name,
)


def test_split_server_prefix_respects_known_prefixes():
    prefixed = "deepwiki-read_wiki_structure"

    unprefixed, server = split_server_prefix_from_name(
        prefixed, known_server_prefixes=["deepwiki", "zapier"]
    )

    assert unprefixed == "read_wiki_structure"
    assert server == "deepwiki"


def test_split_server_prefix_ignores_unknown_prefixes():
    prefixed = "unknown-read_wiki_structure"

    unprefixed, server = split_server_prefix_from_name(
        prefixed, known_server_prefixes=["deepwiki"]
    )

    assert unprefixed == prefixed
    assert server == ""


def test_get_possible_server_name_prefixes_returns_alias_server_name_and_id():
    server = SimpleNamespace(
        alias="alias-name", server_name="server-name", server_id="server-id"
    )

    prefixes = get_possible_server_name_prefixes(server)

    assert prefixes == ["alias-name", "server-name", "server-id"]
