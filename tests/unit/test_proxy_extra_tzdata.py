"""
Static checks that the `proxy` extra ships tzdata.

Legacy IANA aliases such as `Asia/Calcutta` are absent from the OS zone database
on stock Ubuntu 24.04+ (they moved to the separate `tzdata-legacy` package).
zoneinfo then falls back to the `tzdata` wheel, and `pip install litellm[proxy]`
without it exits at proxy startup with
`ZoneInfoNotFoundError: 'tzlocal() does not support non-zoneinfo timezones like
Asia/Calcutta. Please use a timezone in the form of Continent/City'`
(BerriAI/litellm#49813). The Docker images install the OS `tzdata` apk package,
so only the pip surface regresses when this extra loses the wheel
"""

import sys
from pathlib import Path
from typing import Final

from packaging.requirements import Requirement

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

PYPROJECT: Final = Path(__file__).resolve().parents[2] / "pyproject.toml"


def test_proxy_extra_requires_tzdata_on_all_platforms() -> None:
    manifest: Final = tomllib.loads(PYPROJECT.read_text())
    proxy_specs: Final = tuple(Requirement(spec) for spec in manifest["project"]["optional-dependencies"]["proxy"])
    tzdata_specs: Final = tuple(spec for spec in proxy_specs if spec.name == "tzdata")
    assert len(tzdata_specs) == 1, (
        "pip install litellm[proxy] must ship tzdata: proxy startup with a legacy IANA alias like "
        f"TZ=Asia/Calcutta dies with ZoneInfoNotFoundError without it, got specs {proxy_specs}"
    )
    marker: Final = tzdata_specs[0].marker
    assert marker is None, f"tzdata is required on every platform the proxy starts on, found marker {marker}"
