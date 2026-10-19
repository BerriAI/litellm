from collections.abc import Iterator
from io import StringIO
from math import isnan
from pathlib import Path
from typing import Final

import pytest
import yaml

from litellm.rust_bridge.config import INCLUDE_PATH, INCLUDE_PLAN, PARSE_YAML, parse_yaml


@pytest.fixture(params=("native", "python"))
def backend(request: pytest.FixtureRequest) -> Iterator[str]:
    selected: Final[str] = request.param
    if selected == "python":
        PARSE_YAML.override(None)
        INCLUDE_PLAN.override(None)
        INCLUDE_PATH.override(None)
    else:
        pytest.importorskip("litellm.rust_bridge._native")
        assert PARSE_YAML.load() is not None
    yield selected
    PARSE_YAML.reset()
    INCLUDE_PLAN.reset()
    INCLUDE_PATH.reset()


@pytest.mark.parametrize(
    "value",
    (
        "on",
        "OFF",
        "Yes",
        "nO",
        "null",
        "~",
        "",
        "0",
        "-0",
        "012",
        "08",
        "0o12",
        "-0b1_010",
        "+0xA",
        "1_000",
        "1:02:03",
        "1:02:03.5",
        "1000000000000000:12:34.6",
        "1e3",
        "1.0e3",
        "1.0e+3",
        ".5",
        "-.5",
        "+.5",
        ".inf",
        "-.Inf",
        "'on'",
        '"012"',
        "os.environ/UNRESOLVED",
        "2026-01-02",
        "!!binary aGVsbG8=",
        "18446744073709551616",
        "!!str true",
        "!!set {a, b}",
    ),
)
def test_parser_preserves_pyyaml_scalar_values_and_types(backend: str, value: str) -> None:
    source: Final = f"value: {value}\n"
    actual: Final = parse_yaml(source)
    expected: Final = yaml.safe_load(source)
    assert actual == expected
    assert isinstance(actual, dict)
    assert type(actual["value"]) is type(expected["value"])


def test_parser_preserves_duplicate_key_order_and_unwritten_settings(backend: str) -> None:
    source: Final = "extension: null\nzero: 0\ndisabled: false\nprivate: os.environ/UNRESOLVED\nzero: 2\n"
    actual: Final = parse_yaml(source)
    expected: Final = yaml.safe_load(source)
    assert actual == expected
    assert isinstance(actual, dict)
    assert tuple(actual) == tuple(expected)


def test_parser_preserves_shared_and_recursive_yaml_aliases(backend: str) -> None:
    actual: Final = parse_yaml("first: &first {self: *first}\nsecond: *first\n")
    assert isinstance(actual, dict)
    assert actual["first"] is actual["second"]
    assert actual["first"]["self"] is actual["first"]


def test_parser_preserves_merge_keys(backend: str) -> None:
    source: Final = "base: &base {a: 1, b: 2}\nmerged: {<<: *base, b: 3}\n"
    assert parse_yaml(source) == yaml.safe_load(source)


@pytest.mark.parametrize("source", ("value: [", "value: !unknown secret", "---\none\n---\ntwo"))
def test_parser_preserves_yaml_error_types_and_marks(backend: str, source: str) -> None:
    with pytest.raises(yaml.YAMLError) as expected:
        yaml.safe_load(StringIO(source))
    with pytest.raises(type(expected.value)) as actual:
        parse_yaml(StringIO(source))
    assert str(actual.value) == str(expected.value)


@pytest.mark.parametrize("source", ("", "null", "false", "0", "[]", "{}"))
def test_parser_preserves_falsey_root_documents(backend: str, source: str) -> None:
    actual: Final = parse_yaml(source)
    expected: Final = yaml.safe_load(source)
    assert actual == expected
    assert type(actual) is type(expected)


def test_parser_preserves_nan_values(backend: str) -> None:
    actual: Final = parse_yaml("value: .nan")
    assert isinstance(actual, dict)
    assert isinstance(actual["value"], float)
    assert isnan(actual["value"])


@pytest.mark.parametrize(
    "path",
    tuple(sorted((Path(__file__).resolve().parents[3] / "litellm/proxy/example_config_yaml").glob("*.yaml"))),
    ids=lambda path: path.name,
)
def test_native_parser_preserves_existing_proxy_examples(path: Path) -> None:
    pytest.importorskip("litellm.rust_bridge._native")
    native: Final = PARSE_YAML.load()
    assert native is not None
    source: Final = path.read_text()
    supported, actual = native(source)
    expected: Final = yaml.safe_load(source)
    assert supported
    assert actual == expected
    assert isinstance(actual, dict)
    assert tuple(actual) == tuple(expected)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source, error", (("text", AttributeError), ("[]", AttributeError), ("0", TypeError), ("false", TypeError))
)
async def test_proxy_keeps_invalid_root_error_types(
    backend: str, tmp_path: Path, source: str, error: type[Exception]
) -> None:
    from litellm.proxy.proxy_server import ProxyConfig

    path: Final = tmp_path / "config.yaml"
    path.write_text(source)
    with pytest.raises(error):
        await ProxyConfig()._get_config_from_file(str(path))
