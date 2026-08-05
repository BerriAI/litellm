import os
import re
from collections.abc import Iterator

# Define the base directory for the litellm repository and documentation path
repo_base = "./litellm"  # Change this to your actual path

_GETENV_ARGS = r"""\(\s*['"]([^'"]+)['"]\s*(?:,\s*[^)]*)?\)"""
_GET_SECRET_ARGS = r"""\(\s*['"]([^'"]+)['"]\s*(?:,\s*[^)]*|,\s*default_value=[^)]*)?\)"""

ENV_KEY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"os\.getenv" + _GETENV_ARGS),
    re.compile(r"(?<![\w.])(?:litellm\.(?:utils\.)?)?get_secret(?:_str|_bool)?" + _GET_SECRET_ARGS),
)

DOCS_BASE = "./docs/my-website/docs"
REFERENCE_TABLE_PATH = f"{DOCS_BASE}/proxy/config_settings.md"
DOCS_SUFFIXES = (".md", ".mdx")
DOCUMENTED_KEY_PATTERN = re.compile(r"\b[A-Z][A-Z0-9_]*\b")

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
    return frozenset(
        key for file_path in _files_with_suffix(base_dir, (".py",)) for key in extract_env_keys(_read_text(file_path))
    )


def _files_with_suffix(base_dir: str, suffixes: tuple[str, ...]) -> Iterator[str]:
    for root, dirs, files in os.walk(base_dir):
        # Skip dependency/venv directories - prevents picking up env vars from installed packages
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        yield from (os.path.join(root, name) for name in files if name.endswith(suffixes))


def _read_text(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def extract_documented_keys(docs_content: str) -> frozenset[str]:
    """Return every env-var-shaped name mentioned anywhere in a documentation page."""
    return frozenset(DOCUMENTED_KEY_PATTERN.findall(docs_content))


def collect_documented_keys(docs_dir: str) -> frozenset[str]:
    """Return every env-var-shaped name mentioned on any page under ``docs_dir``."""
    return frozenset(
        key
        for file_path in _files_with_suffix(docs_dir, DOCS_SUFFIXES)
        for key in extract_documented_keys(_read_text(file_path))
    )


def undocumented_env_keys(base_dir: str, docs_dir: str) -> frozenset[str]:
    """Return the env vars read under ``base_dir`` that no page under ``docs_dir`` mentions."""
    return collect_env_keys(base_dir) - collect_documented_keys(docs_dir)


def main() -> None:
    if not os.path.isdir(DOCS_BASE):
        raise Exception(f"No documentation found at {DOCS_BASE}; check out BerriAI/litellm-docs into docs/my-website")

    undocumented_keys = undocumented_env_keys(repo_base, DOCS_BASE)
    if undocumented_keys:
        raise Exception(
            f"Environment variables read under {repo_base} but mentioned nowhere in the docs: "
            f"{sorted(undocumented_keys)}"
            f"\nDocument each one, either on the relevant provider page or as a row in the "
            f"'environment variables - Reference' table in {REFERENCE_TABLE_PATH}"
        )
    print(f"Every environment variable read under {repo_base} is documented")


if __name__ == "__main__":
    main()
