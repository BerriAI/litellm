"""
Static checks on docker/Dockerfile.fips.

The FIPS image must not ship the PyPI `cryptography` wheel, which statically
links its own OpenSSL. The builder stage compiles it from source against the
FIPS-configured system libcrypto and asserts the loaded backend reports the
system OpenSSL before anything is copied into the runtime image.
"""

import re
import shlex
from pathlib import Path
from typing import Final

import pytest

DOCKERFILE_PATH: Final = Path(__file__).resolve().parents[2] / "docker" / "Dockerfile.fips"

pytestmark = pytest.mark.skipif(not DOCKERFILE_PATH.exists(), reason="Dockerfile.fips not present in this checkout")


def _builder_stage() -> str:
    match: Final = re.search(
        r"^FROM \S+ AS builder\n(.*?)(?=^FROM )", DOCKERFILE_PATH.read_text(), re.MULTILINE | re.DOTALL
    )
    assert match, "Dockerfile.fips has no `FROM ... AS builder` stage"
    return match.group(1)


def _instructions(stage: str) -> tuple[str, ...]:
    return tuple(" ".join(part.strip() for part in block.split("\\\n")) for block in re.split(r"\n(?=[A-Z]+ )", stage))


def _apk_packages(stage: str) -> frozenset[str]:
    apk_runs: Final = tuple(i for i in _instructions(stage) if i.startswith("RUN") and "apk add" in i)
    assert apk_runs, "builder stage installs nothing with apk"
    return frozenset(
        word for run in apk_runs for word in shlex.split(run.split("apk add", 1)[1]) if not word.startswith("-")
    )


def test_builder_installs_the_toolchain_cryptography_needs_to_compile_against_system_openssl() -> None:
    packages: Final = _apk_packages(_builder_stage())

    assert {"rust", "pkgconf", "openssl-dev"} <= packages, sorted(packages)


def test_builder_forbids_the_cryptography_wheel_before_the_first_uv_sync() -> None:
    instructions: Final = _instructions(_builder_stage())
    first_sync: Final = next(i for i, text in enumerate(instructions) if text.startswith("RUN uv sync"))
    no_binary: Final = tuple(
        value
        for text in instructions[:first_sync]
        if text.startswith("ENV")
        for key, _, value in (token.partition("=") for token in shlex.split(text.removeprefix("ENV")))
        if key == "UV_NO_BINARY_PACKAGE"
    )

    assert no_binary, "no UV_NO_BINARY_PACKAGE set before the first uv sync"
    assert "cryptography" in no_binary[-1].split(","), no_binary


def test_builder_asserts_the_cryptography_backend_reports_the_system_openssl_after_the_last_uv_sync() -> None:
    instructions: Final = _instructions(_builder_stage())
    last_sync: Final = max(i for i, text in enumerate(instructions) if text.startswith("RUN uv sync"))
    assertion: Final = tuple(
        text
        for text in instructions[last_sync:]
        if "cryptography.hazmat.backends.openssl.backend" in text and "openssl_version_text()" in text
    )

    assert len(assertion) == 1, instructions[last_sync:]
    assert "openssl version" in assertion[0] and "grep" in assertion[0], assertion[0]
