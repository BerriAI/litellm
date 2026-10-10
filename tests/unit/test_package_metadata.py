import sys
from pathlib import Path
from typing import Final

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import NormalizedName, canonicalize_name

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
PYPROJECT: Final = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
LOCK: Final = tomllib.loads((PROJECT_ROOT / "uv.lock").read_text())
OPTIONAL_DEPENDENCIES: Final = PYPROJECT["project"]["optional-dependencies"]
AIOHTTP_POOL_POISONING_RANGE: Final = SpecifierSet(">=3.14.0,<3.14.2")
AIOHTTP_POOL_POISONING_RELEASES: Final = ("3.14.0", "3.14.1")
AIOHTTP_POOL_POISONING_SOURCE: Final = "aio-libs/aiohttp#12953, fixed in aiohttp 3.14.2"
CLI_RUNTIME_DEPENDENCIES: Final = frozenset({"rich", "pyyaml", "requests"})
PROXY_SERVER_ONLY_DEPENDENCIES: Final = frozenset(
    {
        "fastapi",
        "uvicorn",
        "gunicorn",
        "granian",
        "starlette",
        "boto3",
        "polars",
        "soundfile",
        "mcp",
        "cryptography",
        "apscheduler",
        "rq",
        "litellm-enterprise",
        "litellm-proxy-extras",
    }
)


def _locked_package(name: str) -> dict[str, object]:
    return next(package for package in LOCK["package"] if canonicalize_name(package["name"]) == name)


def _requirement_names(requirements: list[str]) -> frozenset[NormalizedName]:
    return frozenset(canonicalize_name(Requirement(requirement).name) for requirement in requirements)


def test_importing_litellm_reports_the_declared_release_version() -> None:
    import litellm

    assert litellm._version.version == PYPROJECT["project"]["version"]


def test_every_declared_extra_is_locked_with_its_requirements() -> None:
    declared: Final = {
        canonicalize_name(extra): _requirement_names(requirements)
        for extra, requirements in OPTIONAL_DEPENDENCIES.items()
    }
    locked: Final = _locked_package("litellm")
    locked_extras: Final = {
        canonicalize_name(extra): frozenset(canonicalize_name(entry["name"]) for entry in entries)
        for extra, entries in locked["optional-dependencies"].items()
    }
    provided: Final = frozenset(canonicalize_name(extra) for extra in locked["metadata"]["provides-extras"])
    assert declared
    assert all(declared.values())
    assert frozenset(declared) == provided
    assert declared == locked_extras


def test_cli_extra_is_a_thin_client_install() -> None:
    cli: Final = _requirement_names(OPTIONAL_DEPENDENCIES["cli"])
    assert CLI_RUNTIME_DEPENDENCIES - cli == frozenset()
    assert cli & PROXY_SERVER_ONLY_DEPENDENCIES == frozenset()


def test_declared_aiohttp_floor_excludes_pool_poisoning_releases() -> None:
    declared: Final = tuple(
        Requirement(requirement)
        for requirement in PYPROJECT["project"]["dependencies"]
        if canonicalize_name(Requirement(requirement).name) == "aiohttp"
    )
    assert len(declared) == 1
    assert (
        tuple(release for release in AIOHTTP_POOL_POISONING_RELEASES if declared[0].specifier.contains(release)) == ()
    ), AIOHTTP_POOL_POISONING_SOURCE


def test_locked_aiohttp_version_is_not_pool_poisoning() -> None:
    locked: Final = _locked_package("aiohttp")["version"]
    assert isinstance(locked, str)
    assert not AIOHTTP_POOL_POISONING_RANGE.contains(locked), AIOHTTP_POOL_POISONING_SOURCE
