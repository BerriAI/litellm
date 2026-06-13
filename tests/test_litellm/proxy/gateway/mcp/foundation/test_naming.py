from expression import Result

from litellm.proxy.gateway.mcp.foundation.naming import (
    is_valid_name,
    namespace_tool,
    split_namespaced,
)


def test_namespace_round_trips_through_split():
    name = namespace_tool("github", "create_issue")
    assert name == "github__create_issue"
    r = split_namespaced(name)
    match r:
        case Result(tag="ok", ok=(alias, tool)):
            assert alias == "github"
            assert tool == "create_issue"
        case _:
            raise AssertionError(f"expected ok, got {r}")


def test_split_rejects_non_namespaced():
    r = split_namespaced("nodelimiter")
    match r:
        case Result(tag="error", error=err):
            assert err.tag == "invalid_input"
        case _:
            raise AssertionError(f"expected invalid_input error, got {r}")


def test_split_rejects_invalid_sep986_chars():
    r = split_namespaced("bad name__tool")
    match r:
        case Result(tag="error", error=err):
            assert err.tag == "invalid_input"
        case _:
            raise AssertionError(f"expected invalid_input error, got {r}")


def test_validation_accepts_sep986_and_rejects_others():
    assert is_valid_name("a.b-c_d")
    assert is_valid_name("x" * 128)
    assert not is_valid_name("x" * 129)
    assert not is_valid_name("")
    assert not is_valid_name("has space")
    assert not is_valid_name("slash/name")


def test_split_keeps_only_first_delimiter():
    r = split_namespaced(namespace_tool("srv", "a__b"))
    match r:
        case Result(tag="ok", ok=(alias, tool)):
            assert alias == "srv"
            assert tool == "a__b"
        case _:
            raise AssertionError(f"expected ok, got {r}")
