"""Centralized CLI color theme for LiteLLM proxy output.

Uses `rich` for terminal styling. The theme is selected via the
``LITELLM_CLI_THEME`` environment variable ("dark" or "light").
Defaults to "dark" for bright colors that read well on dark terminal
backgrounds.

Set ``LITELLM_CLI_THEME=light`` for terminals with a light background,
or ``LITELLM_CLI_THEME=none`` to disable all color output.
"""

from __future__ import annotations

import os
import sys
from typing import Final

from rich.console import Console
from rich.style import Style

_CLI_THEME: Final[str] = os.environ.get("LITELLM_CLI_THEME", "dark").lower()

# Dark theme: brighter hues that pop on dark backgrounds
_DARK_STYLES: Final[dict[str, Style]] = {
    "success": Style(color="bright_green", bold=True),
    "warning": Style(color="bright_yellow", bold=True),
    "info": Style(color="bright_blue", bold=True),
    "accent": Style(color="bright_cyan", bold=True),
    "dim": Style(color="grey63"),
    "reset": Style(),
}

# Light theme: darker hues that read well on light backgrounds
_LIGHT_STYLES: Final[dict[str, Style]] = {
    "success": Style(color="green3", bold=True),
    "warning": Style(color="yellow3", bold=True),
    "info": Style(color="blue", bold=True),
    "accent": Style(color="cyan", bold=True),
    "dim": Style(color="grey50"),
    "reset": Style(),
}

_NO_COLOR_STYLES: Final[dict[str, Style]] = {
    "success": Style(),
    "warning": Style(),
    "info": Style(),
    "accent": Style(),
    "dim": Style(),
    "reset": Style(),
}

if _CLI_THEME == "light":
    _STYLES: Final = _LIGHT_STYLES
elif _CLI_THEME == "none":
    _STYLES = _NO_COLOR_STYLES
else:
    _STYLES = _DARK_STYLES

# Console is auto-detected: if stdout is not a TTY, rich disables color automatically.
console: Final[Console] = Console(stderr=False, file=sys.stdout, highlight=False, no_color=sys.stdout.isatty() is False)


def success(message: str) -> None:
    console.print(message, style=_STYLES["success"])


def warning(message: str) -> None:
    console.print(message, style=_STYLES["warning"])


def info(message: str) -> None:
    console.print(message, style=_STYLES["info"])


def accent(message: str) -> None:
    console.print(message, style=_STYLES["accent"])
