from pathlib import Path

import pytest

from litellm.proxy._experimental.mcp_server import openapi_to_mcp_generator as gen


def test_parse_openapi_spec_covers_json_and_safe_yaml_cases() -> None:
    assert gen._parse_openapi_spec('{"openapi":"3.0.0"}') == {"openapi": "3.0.0"}
    assert gen._parse_openapi_spec("openapi: 3.0.0\nrevision: 1\n") == {
        "openapi": "3.0.0",
        "revision": 1,
    }

    with pytest.raises(TypeError, match="OpenAPI spec must be a JSON or YAML object"):
        gen._parse_openapi_spec("42")


@pytest.mark.parametrize(
    "document,error",
    [
        ("<<: {openapi: 3.0.0}\n", "YAML merge keys are not supported"),
        (
            "shared: &shared\ncomponents:\n  schemas:\n    Request: *shared\n",
            "YAML aliases are not supported",
        ),
    ],
)
def test_parse_openapi_spec_rejects_yaml_aliases_and_merges(document: str, error: str) -> None:
    import yaml

    with pytest.raises(yaml.YAMLError, match=error):
        gen._parse_openapi_spec(document)


def test_parse_openapi_spec_rejects_oversized_yaml_integer() -> None:
    import yaml

    oversized_integer = "1" + ":11" * gen._MAX_YAML_INT_LENGTH
    with pytest.raises(yaml.YAMLError, match="YAML integer is too long"):
        gen._parse_openapi_spec(f"openapi: 3.0.0\nx-expensive: {oversized_integer}\n")


@pytest.mark.asyncio
async def test_load_openapi_spec_async_reads_local_yaml(tmp_path: Path) -> None:
    spec_path = tmp_path / "openapi.yaml"
    spec_path.write_text("openapi: 3.0.0\npaths: {}\n", encoding="utf-8")

    assert await gen.load_openapi_spec_async(str(spec_path)) == {
        "openapi": "3.0.0",
        "paths": {},
    }
