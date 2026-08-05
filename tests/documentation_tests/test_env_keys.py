import os
import re
from collections.abc import Iterator

# Define the base directory for the litellm repository and documentation path
repo_base = "./litellm"  # Change this to your actual path

_GETENV_ARGS = r"""\(\s*['"]([^'"]+)['"]\s*(?:,\s*[^)]*)?\)"""
_GET_SECRET_ARGS = r"""\(\s*['"]([^'"]+)['"]\s*(?:,\s*[^)]*|,\s*default_value=[^)]*)?\)"""

ENV_KEY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"os\.getenv" + _GETENV_ARGS),
    re.compile(r"litellm\.get_secret" + _GET_SECRET_ARGS),
    re.compile(r"litellm\.get_secret_str" + _GET_SECRET_ARGS),
    re.compile(r"(?<![\w.])(?:litellm\.)?get_secret_bool" + _GET_SECRET_ARGS),
)

# Terminal/environment detection variables that should not be documented
# These are internal variables used for terminal detection, not user-configurable settings
# Guard-only env vars: read solely to raise on invalid values; the only valid
# value is the default, so there is nothing meaningful to document.
EXCLUDED_GUARD_ONLY_VARS = {
    "MAVVRIK_FOCUS_FREQUENCY",
}

# Temporary/internal rollout flags are intentionally not added to the public
# environment settings docs until the feature is ready for broad use.
EXCLUDED_ROLLOUT_FLAGS = {
    "LITELLM_USE_RUST_OCR",
    "LITELLM_RUST",
}

EXCLUDED_TERMINAL_VARS = {
    "TERM",
    "TERM_PROGRAM",
    "TERM_PROGRAM_VERSION",
    "TERM_SESSION_ID",
    "VTE_VERSION",
    "KITTY_WINDOW_ID",
    "KONSOLE_VERSION",
    "ITERM_PROFILE",
    "ITERM_PROFILE_NAME",
    "ITERM_SESSION_ID",
    "WEZTERM_VERSION",
    "WT_SESSION",
    "GNOME_TERMINAL_SCREEN",
    "ALACRITTY_SOCKET",
}

EXCLUDED_KEYS = frozenset(EXCLUDED_TERMINAL_VARS | EXCLUDED_GUARD_ONLY_VARS | EXCLUDED_ROLLOUT_FLAGS)

# Directories to skip (dependencies, venvs, caches) - only scan litellm source
SKIP_DIRS = {
    ".venv",
    "venv",
    "__pycache__",
    ".git",
    "node_modules",
    "site-packages",
    ".eggs",
    "dist",
    "build",
}


def extract_env_keys(source: str) -> frozenset[str]:
    """Return every documentable env var name read by the given Python source."""
    return frozenset(
        match for pattern in ENV_KEY_PATTERNS for match in pattern.findall(source) if match not in EXCLUDED_KEYS
    )


def collect_env_keys(base_dir: str) -> frozenset[str]:
    """Return every documentable env var name read anywhere under ``base_dir``."""
    return frozenset(key for file_path in _python_files(base_dir) for key in extract_env_keys(_read_text(file_path)))


def _python_files(base_dir: str) -> Iterator[str]:
    for root, dirs, files in os.walk(base_dir):
        # Skip dependency/venv directories - prevents picking up env vars from installed packages
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        yield from (os.path.join(root, name) for name in files if name.endswith(".py"))


def _read_text(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def extract_documented_keys(docs_content: str) -> frozenset[str]:
    """Return the key names listed in the 'environment variables - Reference' table."""
    section = re.search(
        r"### environment variables - Reference(.*?)(?=\n###|\Z)",
        docs_content,
        re.DOTALL | re.MULTILINE,
    )
    if section is None:
        return frozenset()
    # Match | KEY_NAME | description | - capture first column only
    return frozenset(
        match.group(1).strip()
        for match in (re.match(r"^\|\s*([A-Z_][A-Z0-9_]*)\s*\|", line) for line in section.group(1).split("\n"))
        if match is not None
    )


def main() -> None:
    env_keys = collect_env_keys(repo_base)
    print(env_keys)

    docs_path = "./docs/my-website/docs/proxy/config_settings.md"  # Path to the documentation
    try:
        documented_keys = extract_documented_keys(_read_text(docs_path))
    except Exception as e:
        raise Exception(f"Error reading documentation: {e}, \n repo base - {os.listdir('./')}")

    print(f"documented_keys: {documented_keys}")
    undocumented_keys = env_keys - documented_keys

    print("Keys expected in 'environment settings' (found in code):")
    for key in sorted(env_keys):
        print(key)

    if undocumented_keys:
        raise Exception(f"\nKeys not documented in 'environment settings - Reference': {sorted(undocumented_keys)}")
    print(f"\nAll keys are documented in 'environment settings - Reference'. - {env_keys}")


if __name__ == "__main__":
    main()
