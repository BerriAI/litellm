"""Regression: proxy-runtime + USE_DDTRACE must pull a ddtrace that tolerates mcp>=2."""

from __future__ import annotations

import re
from pathlib import Path

import tomllib


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_proxy_runtime_requires_ddtrace_with_mcp2_guard() -> None:
    """ddtrace <4.15 raises on mcp 2.x (BaseSession moved); keep the floor at 4.15.0.

    See https://github.com/BerriAI/litellm/issues/46043 and
    DataDog/dd-trace-py#20134.
    """
    pyproject = tomllib.loads((_repo_root() / "pyproject.toml").read_text())
    runtime_deps = pyproject["project"]["optional-dependencies"]["proxy-runtime"]
    ddtrace_pins = [dep for dep in runtime_deps if dep.startswith("ddtrace")]
    assert ddtrace_pins, "proxy-runtime extra must declare ddtrace"
    pin = ddtrace_pins[0]
    match = re.search(r">=\s*([0-9]+(?:\.[0-9]+)*)", pin)
    assert match, f"expected a lower bound in {pin!r}"
    major, minor, *rest = (int(part) for part in match.group(1).split("."))
    patch = rest[0] if rest else 0
    assert (major, minor, patch) >= (4, 15, 0), pin
