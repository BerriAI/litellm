# ruff: noqa: T201
# flake8: noqa: T201
"""
LiteLLM Interactive Setup Wizard

Guides users through selecting LLM providers, entering API keys,
and generating a proxy config file — mirroring the Claude Code onboarding UX.
"""

import importlib.metadata
import os
import secrets
import sys
import termios
import tty
from pathlib import Path
from typing import Dict, List, Optional, Set

# ---------------------------------------------------------------------------
# Provider definitions
# ---------------------------------------------------------------------------

PROVIDERS = [
    {
        "id": "openai",
        "name": "OpenAI",
        "description": "GPT-4o, GPT-4o-mini, o3-mini",
        "env_key": "OPENAI_API_KEY",
        "key_hint": "sk-...",
        "models": ["gpt-4o", "gpt-4o-mini"],
    },
    {
        "id": "anthropic",
        "name": "Anthropic",
        "description": "Claude Opus 4.6, Sonnet 4.6, Haiku 4.5",
        "env_key": "ANTHROPIC_API_KEY",
        "key_hint": "sk-ant-...",
        "models": ["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
    },
    {
        "id": "gemini",
        "name": "Google Gemini",
        "description": "Gemini 2.0 Flash, Gemini 2.5 Pro",
        "env_key": "GEMINI_API_KEY",
        "key_hint": "AIza...",
        "models": ["gemini/gemini-2.0-flash", "gemini/gemini-2.5-pro"],
    },
    {
        "id": "azure",
        "name": "Azure OpenAI",
        "description": "GPT-4o via Azure",
        "env_key": "AZURE_API_KEY",
        "key_hint": "your-azure-key",
        "models": [],
        "needs_api_base": True,
        "api_base_hint": "https://<resource>.openai.azure.com/",
        "api_version": "2024-07-01-preview",
    },
    {
        "id": "bedrock",
        "name": "AWS Bedrock",
        "description": "Claude 3.5, Llama 3 via AWS",
        "env_key": "AWS_ACCESS_KEY_ID",
        "key_hint": "AKIA...",
        "models": ["bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"],
        "needs_extra": True,
        "extra_keys": ["AWS_SECRET_ACCESS_KEY", "AWS_REGION_NAME"],
        "extra_hints": ["your-secret-key", "us-east-1"],
    },
    {
        "id": "ollama",
        "name": "Ollama",
        "description": "Local models (llama3.2, mistral, etc.)",
        "env_key": None,
        "key_hint": None,
        "models": ["ollama/llama3.2", "ollama/mistral"],
        "api_base": "http://localhost:11434",
    },
]


# ---------------------------------------------------------------------------
# ANSI colour helpers (no external deps needed)
# ---------------------------------------------------------------------------

_ORANGE = "\033[38;2;215;119;87m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_GREEN = "\033[38;2;78;186;101m"
_BLUE = "\033[38;2;177;185;249m"
_GREY = "\033[38;2;153;153;153m"
_RESET = "\033[0m"
_CHECK = "✔"

_CURSOR_HIDE = "\033[?25l"
_CURSOR_SHOW = "\033[?25h"
_MOVE_UP = "\033[{}A"


def _supports_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(code: str, text: str) -> str:
    if _supports_color():
        return f"{code}{text}{_RESET}"
    return text


def orange(t: str) -> str:
    return _c(_ORANGE, t)


def bold(t: str) -> str:
    return _c(_BOLD, t)


def green(t: str) -> str:
    return _c(_GREEN, t)


def blue(t: str) -> str:
    return _c(_BLUE, t)


def grey(t: str) -> str:
    return _c(_GREY, t)


def dim(t: str) -> str:
    return _c(_DIM, t)


# ---------------------------------------------------------------------------
# ASCII art
# ---------------------------------------------------------------------------

LITELLM_ASCII = r"""
  ██╗     ██╗████████╗███████╗██╗     ██╗     ███╗   ███╗
  ██║     ██║╚══██╔══╝██╔════╝██║     ██║     ████╗ ████║
  ██║     ██║   ██║   █████╗  ██║     ██║     ██╔████╔██║
  ██║     ██║   ██║   ██╔══╝  ██║     ██║     ██║╚██╔╝██║
  ███████╗██║   ██║   ███████╗███████╗███████╗██║ ╚═╝ ██║
  ╚══════╝╚═╝   ╚═╝   ╚══════╝╚══════╝╚══════╝╚═╝     ╚═╝
"""

DIVIDER = dim("  " + "╌" * 74)


def _print_welcome() -> None:
    try:
        version = importlib.metadata.version("litellm")
    except Exception:
        version = "unknown"

    print()
    print(orange(LITELLM_ASCII.rstrip("\n")))
    print(f"  {orange('Welcome')} to {bold('LiteLLM')} {grey('v' + version)}")
    print()
    print(DIVIDER)
    print()


# ---------------------------------------------------------------------------
# Arrow-key provider selector
# ---------------------------------------------------------------------------

def _read_key() -> str:
    """Read one keypress from /dev/tty in raw mode."""
    with open("/dev/tty", "rb") as tty_fh:
        fd = tty_fh.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = tty_fh.read(1)
            if ch == b"\x1b":
                ch2 = tty_fh.read(1)
                if ch2 == b"[":
                    ch3 = tty_fh.read(1)
                    return "\x1b[" + ch3.decode("utf-8", errors="replace")
                return "\x1b" + ch2.decode("utf-8", errors="replace")
            return ch.decode("utf-8", errors="replace")
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _render_selector(cursor: int, selected: Set[int], first_render: bool) -> int:
    """Draw (or redraw) the provider list. Returns number of lines printed."""
    lines = []
    lines.append(f"\n  {bold('Add your first model')}\n")
    lines.append(grey("  ↑↓ to navigate · Space to select · Enter to confirm") + "\n")
    lines.append("\n")

    for i, p in enumerate(PROVIDERS):
        arrow = blue("❯") if i == cursor else " "
        bullet = green("◉") if i in selected else grey("○")
        name_str = bold(p["name"]) if i == cursor else p["name"]
        desc_str = grey(p["description"])
        lines.append(f"  {arrow} {bullet} {name_str}  {desc_str}\n")

    lines.append("\n")
    content = "".join(lines)
    line_count = content.count("\n")

    if not first_render and _supports_color():
        sys.stdout.write(_MOVE_UP.format(line_count))

    sys.stdout.write(content)
    sys.stdout.flush()
    return line_count


def _select_providers() -> List[Dict]:
    """Arrow-key multi-select. Falls back to number input if /dev/tty unavailable."""
    try:
        return _select_providers_interactive()
    except (OSError, termios.error):
        return _select_providers_fallback()


def _select_providers_interactive() -> List[Dict]:
    cursor = 0
    selected: Set[int] = set()

    if _supports_color():
        sys.stdout.write(_CURSOR_HIDE)
        sys.stdout.flush()

    try:
        _render_selector(cursor, selected, first_render=True)

        while True:
            key = _read_key()

            if key == "\x1b[A":  # Up
                cursor = (cursor - 1) % len(PROVIDERS)
            elif key == "\x1b[B":  # Down
                cursor = (cursor + 1) % len(PROVIDERS)
            elif key == " ":  # Space — toggle
                if cursor in selected:
                    selected.discard(cursor)
                else:
                    selected.add(cursor)
            elif key in ("\r", "\n"):  # Enter — confirm
                if not selected:
                    selected.add(cursor)  # select highlighted item if nothing chosen
                break
            elif key in ("\x03", "\x04"):  # Ctrl+C / Ctrl+D
                raise KeyboardInterrupt

            _render_selector(cursor, selected, first_render=False)
    finally:
        if _supports_color():
            sys.stdout.write(_CURSOR_SHOW)
            sys.stdout.flush()

    return [PROVIDERS[i] for i in sorted(selected)]


def _select_providers_fallback() -> List[Dict]:
    """Number-based fallback when raw terminal input is unavailable."""
    print()
    print(f"  {bold('Add your first model')}")
    print(grey("  Enter numbers separated by commas (e.g. 1,2). Press Enter to confirm."))
    print()
    for i, p in enumerate(PROVIDERS, 1):
        print(f"  {grey(str(i) + '.')} {bold(p['name'])}  {grey(p['description'])}")
    print()

    selected_nums: List[int] = []
    while True:
        raw = input(f"  {blue('❯')} Provider(s): ").strip()
        if not raw:
            if not selected_nums:
                print(grey("  Please select at least one provider."))
                continue
            break
        try:
            nums = [int(x.strip()) for x in raw.replace(" ", ",").split(",") if x.strip()]
            valid = [n for n in nums if 1 <= n <= len(PROVIDERS)]
            if not valid:
                print(grey(f"  Enter numbers between 1 and {len(PROVIDERS)}."))
                continue
            selected_nums = sorted(set(valid))
            break
        except ValueError:
            print(grey("  Enter numbers separated by commas, e.g. 1,3"))

    return [PROVIDERS[i - 1] for i in selected_nums]


# ---------------------------------------------------------------------------
# API key collection
# ---------------------------------------------------------------------------

def _collect_keys(providers: List[Dict]) -> Dict[str, str]:
    env_vars: Dict[str, str] = {}
    print()
    print(DIVIDER)
    print()
    print(f"  {bold('Enter your API keys')}")
    print(grey("  Keys are stored only in the generated config file."))
    print()

    for p in providers:
        if p["env_key"] is None:
            # Ollama — no key needed
            print(f"  {green(p['name'])}: {grey('no key needed (uses local Ollama)')}")
            continue

        hint = grey(p.get("key_hint", ""))
        key = ""
        while not key:
            key = input(f"  {blue('❯')} {bold(p['name'])} API key {hint}: ").strip()
            if not key:
                print(grey("  Key is required. Leave blank to skip this provider."))
                skip = input(grey("  Skip? (y/N): ")).strip().lower()
                if skip == "y":
                    break

        if key:
            env_vars[p["env_key"]] = key

        # Extra keys (e.g. AWS secret + region)
        if p.get("needs_extra") and key:
            for extra_key, extra_hint in zip(
                p.get("extra_keys", []), p.get("extra_hints", [])
            ):
                val = input(
                    f"  {blue('❯')} {extra_key} {grey(extra_hint)}: "
                ).strip()
                if val:
                    env_vars[extra_key] = val

        # API base for Azure
        if p.get("needs_api_base") and key:
            api_base = input(
                f"  {blue('❯')} Azure endpoint URL {grey(p.get('api_base_hint', ''))}: "
            ).strip()
            if api_base:
                env_vars[f"_LITELLM_AZURE_API_BASE_{p['id'].upper()}"] = api_base

    return env_vars


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------

def _build_config(
    providers: List[Dict],
    env_vars: Dict[str, str],
    port: int,
    master_key: str,
) -> str:
    lines = ["model_list:"]

    for p in providers:
        if not p["models"] and p["id"] == "azure":
            # Azure — add a generic placeholder
            models_to_add = ["azure/gpt-4o"]
        else:
            models_to_add = p["models"]

        for model in models_to_add:
            # User-facing model name (strip provider prefix for display)
            display_name = model.split("/")[-1] if "/" in model else model
            lines.append(f"  - model_name: {display_name}")
            lines.append(f"    litellm_params:")
            lines.append(f"      model: {model}")

            if p["env_key"] and p["env_key"] in env_vars:
                lines.append(f"      api_key: os.environ/{p['env_key']}")

            if p.get("api_base"):
                lines.append(f"      api_base: {p['api_base']}")
            elif p.get("needs_api_base"):
                azure_base_key = f"_LITELLM_AZURE_API_BASE_{p['id'].upper()}"
                if azure_base_key in env_vars:
                    lines.append(f"      api_base: {env_vars.pop(azure_base_key)}")
            if p.get("api_version"):
                lines.append(f"      api_version: {p['api_version']}")

    lines.append("")
    lines.append("general_settings:")
    lines.append(f"  master_key: {master_key}")
    lines.append("")

    # Write env vars inline so the config is self-contained
    real_env_vars = {k: v for k, v in env_vars.items() if not k.startswith("_LITELLM_")}
    if real_env_vars:
        lines.append("environment_variables:")
        for k, v in real_env_vars.items():
            lines.append(f"  {k}: \"{v}\"")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Proxy settings
# ---------------------------------------------------------------------------

def _proxy_settings() -> tuple[int, str]:
    print()
    print(DIVIDER)
    print()
    print(f"  {bold('Proxy settings')}")
    print()

    port_raw = input(f"  {blue('❯')} Port {grey('[4000]')}: ").strip()
    port: int = int(port_raw) if port_raw.isdigit() else 4000

    key_raw = input(
        f"  {blue('❯')} Master key {grey('[auto-generate]')}: "
    ).strip()
    master_key = key_raw if key_raw else f"sk-{secrets.token_urlsafe(32)}"

    return port, master_key


# ---------------------------------------------------------------------------
# Main wizard entrypoint
# ---------------------------------------------------------------------------

def run_setup_wizard() -> Optional[str]:
    """
    Run the interactive setup wizard.

    Returns the path to the generated config file, or None if aborted.
    """
    try:
        _run_wizard()
    except (KeyboardInterrupt, EOFError):
        print(f"\n\n  {grey('Setup cancelled.')}\n")
        return None
    return None  # caller receives path via side effect (printed to stdout)


def _run_wizard() -> None:
    _print_welcome()

    print(f"  {bold('Lets get started.')}")
    print()

    # Step 1: providers
    providers = _select_providers()

    # Step 2: API keys
    env_vars = _collect_keys(providers)

    # Step 3: proxy settings
    port, master_key = _proxy_settings()

    # Step 4: write config
    config_content = _build_config(providers, env_vars, port, master_key)

    config_path = Path(os.getcwd()) / "litellm_config.yaml"
    config_path.write_text(config_content)

    # Step 5: print success
    print()
    print(DIVIDER)
    print()
    print(f"  {green(_CHECK + ' Config saved')} → {bold(str(config_path))}")
    print()
    print(f"  {bold('To start your proxy:')}")
    print()
    print(f"    {grey('$')} litellm --config {config_path}")
    print()
    print(f"  {bold('Then set your client:')}")
    print()
    print(f"    export OPENAI_BASE_URL=http://localhost:{port}")
    print(f"    export OPENAI_API_KEY={master_key}")
    print()
    print(DIVIDER)
    print()

    # Step 6: offer to start now
    start = input(f"  {blue('❯')} Start the proxy now? {grey('(Y/n)')}: ").strip().lower()
    if start in ("", "y", "yes"):
        print()
        print(f"  {green(_CHECK)} Starting LiteLLM proxy on port {bold(str(port))}…")
        print()
        # exec replaces this process with the proxy server
        os.execlp(  # noqa: S606
            sys.executable,
            sys.executable,
            "-m",
            "litellm",
            "--config",
            str(config_path),
            "--port",
            str(port),
        )
    else:
        print()
        print(
            f"  Run {bold(f'litellm --config {config_path}')} whenever you're ready."
        )
        print()
