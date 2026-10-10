import os
import sys
from pathlib import Path
from typing import Final

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

PROJECT_ROOT: Final = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PYPROJECT: Final = tomllib.loads(Path(PROJECT_ROOT, "pyproject.toml").read_text())
LOCK: Final = tomllib.loads(Path(PROJECT_ROOT, "uv.lock").read_text())


def _locked_root_package() -> dict[str, object]:
    return next(package for package in LOCK["package"] if package["name"] == "litellm")


def test_importing_litellm_reports_the_declared_release_version() -> None:
    import litellm

    assert litellm._version.version == PYPROJECT["project"]["version"]


def test_every_declared_extra_is_locked_with_its_requirements() -> None:
    declared: Final = {
        canonicalize_name(extra): frozenset(canonicalize_name(Requirement(item).name) for item in requirements)
        for extra, requirements in PYPROJECT["project"]["optional-dependencies"].items()
    }
    locked: Final = _locked_root_package()
    locked_extras: Final = {
        canonicalize_name(extra): frozenset(canonicalize_name(entry["name"]) for entry in entries)
        for extra, entries in locked["optional-dependencies"].items()
    }
    provided: Final = tuple(sorted(canonicalize_name(extra) for extra in locked["metadata"]["provides-extras"]))
    assert tuple(sorted(declared)) == provided
    assert declared == locked_extras


def test_cli_extra_is_a_thin_client_install():
    """The `cli` extra must install a working `lite` client without dragging in the
    proxy server runtime. It therefore has to declare the CLI's real third-party
    deps (rich, pyyaml, requests) and must never contain a server-only dependency
    from the `proxy` extra; a leak there silently re-bloats the laptop install.
    """
    import pathlib

    from packaging.requirements import Requirement

    import litellm

    try:
        import tomllib as tomli
    except ImportError:
        try:
            import tomli
        except ImportError:
            pytest.skip("tomli/tomllib not available - skipping dependency check")

    pyproject_path = pathlib.Path(litellm.__file__).parent.parent / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        optional_deps = tomli.load(f)["project"]["optional-dependencies"]

    assert "cli" in optional_deps, "Expected a `cli` extra for the thin lite install"

    cli_names = {Requirement(req).name.lower() for req in optional_deps["cli"]}

    missing = {"rich", "pyyaml", "requests"} - cli_names
    assert not missing, f"`cli` extra is missing deps the lite CLI imports: {missing}"

    server_only = {
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
    leaked = cli_names & server_only
    assert not leaked, f"`cli` extra leaks proxy-server deps onto laptops: {leaked}"


AIOHTTP_POOL_POISONING_RANGE = ">=3.14.0,<3.14.2"
AIOHTTP_POOL_POISONING_RELEASES = ("3.14.0", "3.14.1")


def _load_toml(path):
    try:
        import tomllib as tomli
    except ImportError:
        try:
            import tomli
        except ImportError:
            pytest.skip("tomli/tomllib not available - skipping dependency check")

    with open(path, "rb") as f:
        return tomli.load(f)


def _declared_aiohttp_specifier():
    from packaging.requirements import Requirement

    pyproject = _load_toml(os.path.join(PROJECT_ROOT, "pyproject.toml"))
    for requirement in pyproject["project"]["dependencies"]:
        parsed = Requirement(requirement)
        if parsed.name.lower() == "aiohttp":
            return parsed.specifier
    pytest.fail("aiohttp is no longer a declared runtime dependency of litellm")


def _locked_aiohttp_version():
    lock = _load_toml(os.path.join(PROJECT_ROOT, "uv.lock"))
    for package in lock["package"]:
        if package["name"].lower() == "aiohttp":
            return package["version"]
    pytest.fail("aiohttp is missing from uv.lock")


def test_declared_aiohttp_floor_excludes_pool_poisoning_releases():
    """aiohttp 3.14.0/3.14.1 re-arm the sock_read timer on a keep-alive connection
    after it is back in the idle pool, so the next request to reuse it fails
    instantly with a bogus timeout (aio-libs/aiohttp#12953, fixed in 3.14.2).

    The wheel's own metadata is what pip resolves against, so the floor declared
    here - not just the lockfile - has to exclude that range.
    """
    specifier = _declared_aiohttp_specifier()

    admitted = [v for v in AIOHTTP_POOL_POISONING_RELEASES if specifier.contains(v)]
    assert not admitted, (
        f"litellm declares aiohttp{specifier}, which still admits {admitted}. "
        "Those releases poison pooled keep-alive connections and cause "
        "cross-provider sub-millisecond 'Connection timed out' failures; "
        "keep the floor at >=3.14.2."
    )


def test_locked_aiohttp_version_is_not_pool_poisoning():
    """uv.lock is what the published Docker images install (uv sync --frozen), so a
    lock that drifts back onto 3.14.0/3.14.1 ships the regression regardless of
    what pyproject.toml declares.
    """
    from packaging.specifiers import SpecifierSet

    locked = _locked_aiohttp_version()

    assert not SpecifierSet(AIOHTTP_POOL_POISONING_RANGE).contains(locked), (
        f"uv.lock resolves aiohttp {locked}, which is inside the pool-poisoning "
        f"range {AIOHTTP_POOL_POISONING_RANGE} (aio-libs/aiohttp#12953). "
        "Re-run `uv lock` against an aiohttp>=3.14.2 floor."
    )
