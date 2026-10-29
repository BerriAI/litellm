from collections.abc import Iterator
from pathlib import Path
from typing import Final

import yaml

CONFIG: Final = Path(__file__).resolve().parents[2] / ".circleci" / "config.yml"


def _step_name(step: object) -> str:
    if isinstance(step, str):
        return step
    if isinstance(step, dict) and len(step) == 1:
        return str(next(iter(step)))
    return ""


def _step_names(node: object) -> Iterator[str]:
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "steps" and isinstance(value, list):
                yield from map(_step_name, value)
            yield from _step_names(value)
    if isinstance(node, list):
        for item in node:
            yield from _step_names(item)


def test_every_clone_goes_over_https_because_the_project_has_no_ssh_checkout_key() -> None:
    names: Final = tuple(_step_names(yaml.safe_load(CONFIG.read_text())))
    assert "checkout_over_https" in names
    assert "checkout" not in names
