"""Fixtures for litellm.harness end-to-end tests.

These run the real harness runtimes (claude, codex, opencode, deepagents) against a
real LiteLLM AI Gateway. They skip unless LITELLM_PROXY_API_BASE and
LITELLM_PROXY_API_KEY are set. Model names are gateway model groups and can be
overridden per harness with HARNESS_E2E_MODEL_<HARNESS>.
"""

import os
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from litellm.harness import Gateway, Harness

GATEWAY_BASE = os.environ.get("LITELLM_PROXY_API_BASE", "").strip()
GATEWAY_KEY = os.environ.get("LITELLM_PROXY_API_KEY", "").strip()

DEFAULT_MODELS = {
    Harness.CLAUDE_CODE: "claude-haiku-4-5-20251001",
    Harness.CODEX: "bedrock_mantle/openai.gpt-5.4",
    Harness.OPENCODE: "claude-haiku-4-5-20251001",
    Harness.DEEPAGENTS: "claude-haiku-4-5-20251001",
}

BINARIES = {
    Harness.CLAUDE_CODE: "claude",
    Harness.CODEX: "codex",
    Harness.OPENCODE: "opencode",
}

requires_gateway = pytest.mark.skipif(
    not (GATEWAY_BASE and GATEWAY_KEY),
    reason="LITELLM_PROXY_API_BASE / LITELLM_PROXY_API_KEY not set",
)


def model_for(harness: Harness) -> str:
    override = os.environ.get(f"HARNESS_E2E_MODEL_{harness.name}", "").strip()
    return override or DEFAULT_MODELS[harness]


def harness_available(harness: Harness) -> bool:
    if harness is Harness.DEEPAGENTS:
        try:
            import deepagents  # noqa: F401
            import langchain_litellm  # noqa: F401
        except ImportError:
            return False
        return True
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
        for h in Harness
    ]


@pytest.fixture
def gateway() -> Gateway:
    return Gateway(api_base=GATEWAY_BASE, api_key=GATEWAY_KEY)


@pytest.fixture
def workspace(tmp_path: Path) -> Iterator[Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("# demo\n")
    yield repo
