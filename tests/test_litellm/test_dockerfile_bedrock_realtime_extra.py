"""
Static checks that every proxy Docker image installs the `bedrock-realtime` extra.

Bedrock Nova Sonic speech-to-speech (`/v1/realtime`) needs `aws-sdk-bedrock-runtime`,
which only ships in the `bedrock-realtime` extra. An image whose `uv sync` stages
omit the extra fails every Nova Sonic realtime session with
"Missing aws_sdk_bedrock_runtime for Bedrock realtime".
"""

import os
import re
import sys
from typing import Final

import pytest

from litellm.constants import BEDROCK_REALTIME_SDK_DISTRIBUTION, BEDROCK_REALTIME_SDK_SUPPORTED_RANGE

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

REPO_ROOT: Final = os.path.join(os.path.dirname(__file__), "..", "..")

PROXY_DOCKERFILES: Final = (
    "Dockerfile",
    os.path.join("docker", "Dockerfile.non_root"),
    os.path.join("docker", "Dockerfile.database"),
    os.path.join("gateway", "Dockerfile"),
)

CONTINUED_LINE_RE: Final = re.compile(r"(?:\\\n|[^\n])+")
UV_SYNC_BOUNDARY_RE: Final = re.compile(r"(?=uv sync)")


def _uv_sync_invocations(dockerfile_text: str) -> tuple[str, ...]:
    """Return each `uv sync ...` command, split apart when one RUN holds several (if/else branches)."""
    return tuple(
        part
        for line in CONTINUED_LINE_RE.finditer(dockerfile_text)
        for part in UV_SYNC_BOUNDARY_RE.split(line.group(0))
        if part.startswith("uv sync")
    )


@pytest.mark.parametrize("relative_path", PROXY_DOCKERFILES)
def test_every_uv_sync_installs_bedrock_realtime_extra(relative_path: str):
    dockerfile_path: Final = os.path.join(REPO_ROOT, relative_path)
    if not os.path.exists(dockerfile_path):
        pytest.skip(f"{relative_path} not present in this checkout")

    with open(dockerfile_path, "r", encoding="utf-8") as f:
        contents: Final = f.read()

    invocations: Final = _uv_sync_invocations(contents)
    assert invocations, f"{relative_path} has no `uv sync` invocation"

    missing: Final = tuple(invocation for invocation in invocations if "--extra bedrock-realtime" not in invocation)
    assert not missing, (
        f"{relative_path}: {len(missing)} of {len(invocations)} `uv sync` invocations omit "
        "`--extra bedrock-realtime`, so aws-sdk-bedrock-runtime is absent and Bedrock Nova Sonic "
        "/v1/realtime sessions fail with 'Missing aws_sdk_bedrock_runtime'"
    )


def test_bedrock_realtime_extra_pins_the_range_named_in_the_runtime_error():
    with open(os.path.join(REPO_ROOT, "pyproject.toml"), "rb") as f:
        extra_specs: Final = tomllib.load(f)["project"]["optional-dependencies"]["bedrock-realtime"]

    sdk_specs: Final = tuple(spec for spec in extra_specs if spec.startswith(BEDROCK_REALTIME_SDK_DISTRIBUTION))
    assert len(sdk_specs) == 1, f"expected exactly one {BEDROCK_REALTIME_SDK_DISTRIBUTION} spec, got {extra_specs}"
    requirement: Final = sdk_specs[0].split(";")[0].strip()
    assert requirement == f"{BEDROCK_REALTIME_SDK_DISTRIBUTION}[awscrt]{BEDROCK_REALTIME_SDK_SUPPORTED_RANGE}", (
        f"pyproject pins {requirement!r} but the handler's install hint names "
        f"{BEDROCK_REALTIME_SDK_SUPPORTED_RANGE!r} with the awscrt extra; keep them in sync"
    )
