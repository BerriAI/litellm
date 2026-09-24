import re
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

    assert re.search(
        r"^\s*recursive_type_depth\s*=\s*-1\s*$",
        generator_settings,
        flags=re.MULTILINE,
    )
