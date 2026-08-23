from pathlib import Path
from typing import Final

import pytest


REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[3]
SCHEMA_PATHS: Final = (
    REPOSITORY_ROOT / "schema.prisma",
    REPOSITORY_ROOT / "litellm/proxy/schema.prisma",
    REPOSITORY_ROOT / "litellm-proxy-extras/litellm_proxy_extras/schema.prisma",
)


@pytest.mark.parametrize("schema_path", SCHEMA_PATHS)
def test_prisma_client_uses_recursive_types(schema_path: Path) -> None:
    schema: Final = schema_path.read_text(encoding="utf-8")
    generator_settings: Final = schema.partition("generator client {")[2].partition("}")[0]

    assert "recursive_type_depth = -1" in tuple(line.strip() for line in generator_settings.splitlines())
