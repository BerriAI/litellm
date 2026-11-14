"""Fixtures for the litellm.agent() end-to-end tests.

These run the real harness runtimes (claude, codex, opencode, pi, deepagents) against a real
LiteLLM AI Gateway, routed with the `litellm_proxy/` model prefix. They skip unless
LITELLM_PROXY_API_BASE and LITELLM_PROXY_API_KEY are set. Model groups can be overridden
per harness with HARNESS_E2E_MODEL_<HARNESS>.
"""

import importlib.util
import os
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from litellm import Harness

GATEWAY_BASE = os.environ.get("LITELLM_PROXY_API_BASE", "").strip()
GATEWAY_KEY = os.environ.get("LITELLM_PROXY_API_KEY", "").strip()

DEFAULT_MODEL_GROUPS = {
    Harness.CLAUDE_CODE: "claude-haiku-4-5-20251001",
    Harness.CODEX: "bedrock_mantle/openai.gpt-5.4",
    Harness.OPENCODE: "claude-haiku-4-5-20251001",
    Harness.PI: "claude-haiku-4-5-20251001",
    Harness.DEEPAGENTS: "claude-haiku-4-5-20251001",
}

BINARIES = {
    Harness.CLAUDE_CODE: "claude",
    Harness.CODEX: "codex",
    Harness.OPENCODE: "opencode",
    Harness.PI: "pi",
}

requires_gateway = pytest.mark.skipif(
    not (GATEWAY_BASE and GATEWAY_KEY),
    reason="LITELLM_PROXY_API_BASE / LITELLM_PROXY_API_KEY not set",
)


def model_for(harness: Harness) -> str:
    """`litellm_proxy/<group>`: every model call goes through the gateway."""
    override = os.environ.get(f"HARNESS_E2E_MODEL_{harness.name}", "").strip()
    return f"litellm_proxy/{override or DEFAULT_MODEL_GROUPS[harness]}"


def harness_available(harness: Harness) -> bool:
    if harness is Harness.DEEPAGENTS:
        return all(
            importlib.util.find_spec(m) is not None
            for m in ("deepagents", "langchain_litellm")
        )
    return shutil.which(BINARIES[harness]) is not None


def harness_params() -> list:
    return [
        pytest.param(
            h,
            id=h.value,
            marks=pytest.mark.skipif(
                not harness_available(h), reason=f"{h.value} runtime not installed"
            ),
        )
        for h in DEFAULT_MODEL_GROUPS
    ]


@pytest.fixture
def workspace(tmp_path: Path) -> Iterator[Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("# demo\n")
    yield repo
