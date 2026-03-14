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
import sysconfig
import termios
import tty
from pathlib import Path
from typing import Dict, List, Optional, Set

from litellm.utils import check_valid_key

# ---------------------------------------------------------------------------
# Provider definitions
# ---------------------------------------------------------------------------
# Each entry describes one provider card shown in the wizard.
# `env_key`      — primary env var name (None = no key needed, e.g. Ollama)
# `test_model`   — model passed to check_valid_key for credential validation
#                  (None = skip validation, e.g. Azure needs a deployment name)
# `models`       — default models written into the generated config
# ---------------------------------------------------------------------------

PROVIDERS: List[Dict] = [
    {
        "id": "openai",
        "name": "OpenAI",
        "description": "GPT-4o, GPT-4o-mini, o3-mini",
        "env_key": "OPENAI_API_KEY",
        "key_hint": "sk-...",
        "test_model": "gpt-4o-mini",
        "models": ["gpt-4o", "gpt-4o-mini"],
    },
    {
        "id": "anthropic",
        "name": "Anthropic",
        "description": "Claude Opus 4.6, Sonnet 4.6, Haiku 4.5",
        "env_key": "ANTHROPIC_API_KEY",
        "key_hint": "sk-ant-...",
        "test_model": "claude-haiku-4-5-20251001",
        "models": ["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
    },
    {
        "id": "gemini",
        "name": "Google Gemini",
        "description": "Gemini 2.0 Flash, Gemini 2.5 Pro",
        "env_key": "GEMINI_API_KEY",
        "key_hint": "AIza...",
        "test_model": "gemini/gemini-2.0-flash",
        "models": ["gemini/gemini-2.0-flash", "gemini/gemini-2.5-pro"],
    },
    {
        "id": "azure",
        "name": "Azure OpenAI",
        "description": "GPT-4o via Azure",
        "env_key": "AZURE_API_KEY",
        "key_hint": "your-azure-key",
        "test_model": None,  # needs deployment name — skip validation
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
        "test_model": None,  # multi-key auth — skip validation
        "models": ["bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"],
        "extra_keys": ["AWS_SECRET_ACCESS_KEY", "AWS_REGION_NAME"],
        "extra_hints": ["your-secret-key", "us-east-1"],
    },
    {
        "id": "ollama",
        "name": "Ollama",
        "description": "Local models (llama3.2, mistral, etc.)",
        "env_key": None,
        "key_hint": None,
        "test_model": None,  # local — no remote validation
        "models": ["ollama/llama3.2", "ollama/mistral"],
        "api_base": "http://localhost:11434",
    },
]


# ---------------------------------------------------------------------------
# ANSI colour helpers
# ---------------------------------------------------------------------------

_ORANGE = "\033[38;2;215;119;87m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_GREEN = "\033[38;2;78;186;101m"
_BLUE = "\033[38;2;177;185;249m"
_GREY = "\033[38;2;153;153;153m"
_RESET = "\033[0m"
_CHECK = "✔"
_CROSS = "✘"

_CURSOR_HIDE = "\033[?25l"
_CURSOR_SHOW = "\033[?25h"
_MOVE_UP = "\033[{}A"


def _supports_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(code: str, text: str) -> str:
    return f"{code}{text}{_RESET}" if _supports_color() else text


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
# Layout constants
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


# ---------------------------------------------------------------------------
# Setup wizard
# ---------------------------------------------------------------------------

class SetupWizard:
    """
    Interactive onboarding wizard: provider selection → API keys → config file.

    All methods are static — the class is purely a namespace with clear
    single-responsibility sections. Entry point: SetupWizard.run().
    """

    # ── entry point ─────────────────────────────────────────────────────────

    @staticmethod
    def run() -> None:
        try:
            SetupWizard._wizard()
        except (KeyboardInterrupt, EOFError):
            print(f"\n\n  {grey('Setup cancelled.')}\n")

    # ── wizard steps ────────────────────────────────────────────────────────

    @staticmethod
    def _wizard() -> None:
        SetupWizard._print_welcome()
        print(f"  {bold('Lets get started.')}")
        print()

        providers = SetupWizard._select_providers()
        env_vars = SetupWizard._collect_keys(providers)
        port, master_key = SetupWizard._proxy_settings()

        config_path = Path(os.getcwd()) / "litellm_config.yaml"
        config_path.write_text(
            SetupWizard._build_config(providers, env_vars, port, master_key)
        )

        SetupWizard._print_success(config_path, port, master_key)
        SetupWizard._offer_start(config_path, port, master_key)

    # ── welcome ─────────────────────────────────────────────────────────────

    @staticmethod
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

    # ── provider selector ───────────────────────────────────────────────────

    @staticmethod
    def _select_providers() -> List[Dict]:
        """Arrow-key multi-select. Falls back to number input if /dev/tty unavailable."""
        try:
            return SetupWizard._select_interactive()
        except (OSError, termios.error):
            return SetupWizard._select_fallback()

    @staticmethod
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

    @staticmethod
    def _render_selector(cursor: int, selected: Set[int], first_render: bool) -> int:
        """Draw or redraw the provider list. Returns the number of lines printed."""
        lines = [
            f"\n  {bold('Add your first model')}\n",
            grey("  ↑↓ to navigate · Space to select · Enter to confirm") + "\n",
            "\n",
        ]
        for i, p in enumerate(PROVIDERS):
            arrow = blue("❯") if i == cursor else " "
            bullet = green("◉") if i in selected else grey("○")
            name_str = bold(p["name"]) if i == cursor else p["name"]
            lines.append(f"  {arrow} {bullet} {name_str}  {grey(p['description'])}\n")
        lines.append("\n")

        content = "".join(lines)
        if not first_render and _supports_color():
            sys.stdout.write(_MOVE_UP.format(content.count("\n")))
        sys.stdout.write(content)
        sys.stdout.flush()
        return content.count("\n")

    @staticmethod
    def _select_interactive() -> List[Dict]:
        cursor, selected = 0, set()

        if _supports_color():
            sys.stdout.write(_CURSOR_HIDE)
            sys.stdout.flush()
        try:
            SetupWizard._render_selector(cursor, selected, first_render=True)
            while True:
                key = SetupWizard._read_key()
                if key == "\x1b[A":
                    cursor = (cursor - 1) % len(PROVIDERS)
                elif key == "\x1b[B":
                    cursor = (cursor + 1) % len(PROVIDERS)
                elif key == " ":
                    selected.symmetric_difference_update({cursor})
                elif key in ("\r", "\n"):
                    if not selected:
                        selected.add(cursor)
                    break
                elif key in ("\x03", "\x04"):
                    raise KeyboardInterrupt
                SetupWizard._render_selector(cursor, selected, first_render=False)
        finally:
            if _supports_color():
                sys.stdout.write(_CURSOR_SHOW)
                sys.stdout.flush()

        return [PROVIDERS[i] for i in sorted(selected)]

    @staticmethod
    def _select_fallback() -> List[Dict]:
        """Number-based fallback when raw terminal input is unavailable."""
        print()
        print(f"  {bold('Add your first model')}")
        print(grey("  Enter numbers separated by commas (e.g. 1,2). Press Enter to confirm."))
        print()
        for i, p in enumerate(PROVIDERS, 1):
            print(f"  {grey(str(i) + '.')} {bold(p['name'])}  {grey(p['description'])}")
        print()

        while True:
            raw = input(f"  {blue('❯')} Provider(s): ").strip()
            if not raw:
                print(grey("  Please select at least one provider."))
                continue
            try:
                nums = [int(x.strip()) for x in raw.replace(" ", ",").split(",") if x.strip()]
                valid = sorted({n for n in nums if 1 <= n <= len(PROVIDERS)})
                if not valid:
                    print(grey(f"  Enter numbers between 1 and {len(PROVIDERS)}."))
                    continue
                return [PROVIDERS[i - 1] for i in valid]
            except ValueError:
                print(grey("  Enter numbers separated by commas, e.g. 1,3"))

    # ── key collection ───────────────────────────────────────────────────────

    @staticmethod
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
                print(f"  {green(p['name'])}: {grey('no key needed (uses local Ollama)')}")
                continue

            key = SetupWizard._prompt_key(p)
            if not key:
                continue

            env_vars[p["env_key"]] = key

            for extra_key, extra_hint in zip(
                p.get("extra_keys", []), p.get("extra_hints", [])
            ):
                val = input(f"  {blue('❯')} {extra_key} {grey(extra_hint)}: ").strip()
                if val:
                    env_vars[extra_key] = val

            if p.get("needs_api_base"):
                api_base = input(
                    f"  {blue('❯')} Azure endpoint URL {grey(p.get('api_base_hint', ''))}: "
                ).strip()
                if api_base:
                    env_vars[f"_LITELLM_AZURE_API_BASE_{p['id'].upper()}"] = api_base

            SetupWizard._validate_and_report(p, key)

        return env_vars

    @staticmethod
    def _prompt_key(provider: Dict) -> str:
        """Prompt for a provider's API key, with skip option. Returns the key or ''."""
        hint = grey(provider.get("key_hint", ""))
        while True:
            key = input(f"  {blue('❯')} {bold(provider['name'])} API key {hint}: ").strip()
            if key:
                return key
            print(grey("  Key is required. Leave blank to skip this provider."))
            if input(grey("  Skip? (y/N): ")).strip().lower() == "y":
                return ""

    @staticmethod
    def _validate_and_report(provider: Dict, api_key: str) -> None:
        """
        Validate credentials using litellm.utils.check_valid_key.
        Offers a re-entry loop on failure.
        """
        test_model: Optional[str] = provider.get("test_model")
        if not test_model:
            return  # Azure / Bedrock / Ollama — skip

        while True:
            print(f"  {grey('Testing credentials…')}", end="", flush=True)
            valid = check_valid_key(model=test_model, api_key=api_key)
            if valid:
                print(f"\r  {green(_CHECK + ' ' + provider['name'])} credentials valid        ")
                return

            print(f"\r  {_c(_BOLD, _CROSS)} {bold(provider['name'])} {grey('invalid API key')}")
            if input(f"  {blue('❯')} Re-enter key? {grey('(y/N)')}: ").strip().lower() != "y":
                return

            hint = grey(provider.get("key_hint", ""))
            new_key = input(
                f"  {blue('❯')} {bold(provider['name'])} API key {hint}: "
            ).strip()
            if not new_key:
                return
            api_key = new_key

    # ── proxy settings ───────────────────────────────────────────────────────

    @staticmethod
    def _proxy_settings() -> "tuple[int, str]":
        print()
        print(DIVIDER)
        print()
        print(f"  {bold('Proxy settings')}")
        print()
        port_raw = input(f"  {blue('❯')} Port {grey('[4000]')}: ").strip()
        port = int(port_raw) if port_raw.isdigit() else 4000
        key_raw = input(f"  {blue('❯')} Master key {grey('[auto-generate]')}: ").strip()
        master_key = key_raw if key_raw else f"sk-{secrets.token_urlsafe(32)}"
        return port, master_key

    # ── config generation ────────────────────────────────────────────────────

    @staticmethod
    def _build_config(
        providers: List[Dict],
        env_vars: Dict[str, str],
        port: int,
        master_key: str,
    ) -> str:
        lines = ["model_list:"]
        for p in providers:
            models = p["models"] if p["models"] else (["azure/gpt-4o"] if p["id"] == "azure" else [])
            for model in models:
                display = model.split("/")[-1] if "/" in model else model
                lines += [
                    f"  - model_name: {display}",
                    f"    litellm_params:",
                    f"      model: {model}",
                ]
                if p["env_key"] and p["env_key"] in env_vars:
                    lines.append(f"      api_key: os.environ/{p['env_key']}")
                if p.get("api_base"):
                    lines.append(f"      api_base: {p['api_base']}")
                elif p.get("needs_api_base"):
                    azure_key = f"_LITELLM_AZURE_API_BASE_{p['id'].upper()}"
                    if azure_key in env_vars:
                        lines.append(f"      api_base: {env_vars.pop(azure_key)}")
                if p.get("api_version"):
                    lines.append(f"      api_version: {p['api_version']}")

        lines += ["", "general_settings:", f"  master_key: {master_key}", ""]

        real_vars = {k: v for k, v in env_vars.items() if not k.startswith("_LITELLM_")}
        if real_vars:
            lines.append("environment_variables:")
            for k, v in real_vars.items():
                lines.append(f'  {k}: "{v}"')
            lines.append("")

        return "\n".join(lines)

    # ── success + launch ─────────────────────────────────────────────────────

    @staticmethod
    def _print_success(config_path: Path, port: int, master_key: str) -> None:
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

    @staticmethod
    def _offer_start(config_path: Path, port: int, master_key: str) -> None:
        start = input(f"  {blue('❯')} Start the proxy now? {grey('(Y/n)')}: ").strip().lower()
        if start not in ("", "y", "yes"):
            print()
            print(f"  Run {bold(f'litellm --config {config_path}')} whenever you're ready.")
            print()
            print(grey(f"  Quick test once running:  curl http://localhost:{port}/health"))
            print()
            return

        print()
        print(DIVIDER)
        print()
        print(f"  {bold('Proxy is starting on')} http://localhost:{port}")
        print()
        print(grey("  Your proxy is OpenAI-compatible. Point any OpenAI SDK at it:"))
        print()
        print(f"    export OPENAI_BASE_URL=http://localhost:{port}")
        print(f"    export OPENAI_API_KEY={master_key}")
        print()
        print(grey("  Quick test (in another terminal):"))
        print()
        print(f"    curl http://localhost:{port}/health")
        print()
        print(grey("  Dashboard:"))
        print()
        print(f"    http://localhost:{port}/ui  {grey('(login with your master key)')}")
        print()
        print(DIVIDER)
        print()
        print(f"  {green(_CHECK)} Starting…  {grey('(Ctrl+C to stop)')}")
        print()

        scripts_dir = sysconfig.get_path("scripts")
        litellm_bin = os.path.join(scripts_dir, "litellm")
        os.execlp(litellm_bin, litellm_bin, "--config", str(config_path), "--port", str(port))  # noqa: S606


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def run_setup_wizard() -> Optional[str]:
    """Run the interactive setup wizard. Called by `litellm --setup`."""
    SetupWizard.run()
    return None
