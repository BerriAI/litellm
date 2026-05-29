"""
Regression tests for LIT-2723 - pin the UI build output layout and Node version
so committed `litellm/proxy/_experimental/out/` diffs stop flipping between
`<route>.html` and `<route>/index.html`, and so the build does not silently
fall back to a non-v20 Node.

These are file-level assertions on the build config + script. They do not run
npm or restructure the export; they only guard the locking lines.
"""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
NEXT_CONFIG = REPO_ROOT / "ui" / "litellm-dashboard" / "next.config.mjs"
BUILD_UI_SH = REPO_ROOT / "ui" / "litellm-dashboard" / "build_ui.sh"


def test_next_config_pins_trailing_slash_true():
    """next.config.mjs must declare `trailingSlash: true` so every build emits
    the `<route>/index.html` directory-index layout. Flipping back to the
    `<route>.html` form rewrites every file in `_experimental/out` and breaks
    deployments that rely on directory-index routing (e.g. the MCP OAuth
    callback)."""
    assert NEXT_CONFIG.is_file(), f"next.config.mjs missing at {NEXT_CONFIG}"
    text = NEXT_CONFIG.read_text()
    assert "trailingSlash: true" in text or "trailingSlash:true" in text, (
        "next.config.mjs must set `trailingSlash: true` (LIT-2723). "
        "Removing it makes the static export flip back to the bare "
        "<route>.html form on the next local build."
    )


def test_build_ui_sh_installs_node_v20_and_verifies():
    """build_ui.sh must install v20 (not just `nvm use`), error-check both
    steps, and verify `node -v` actually reports v20 before running the
    build. Without these guards the script silently succeeds against any
    Node version on PATH in non-interactive shells, and the committed
    artifacts pick up the wrong Node."""
    assert BUILD_UI_SH.is_file(), f"build_ui.sh missing at {BUILD_UI_SH}"
    text = BUILD_UI_SH.read_text()

    assert "nvm install v20" in text, (
        "build_ui.sh must `nvm install v20` before `nvm use v20` - "
        "`use` alone silently no-ops when v20 is not already installed."
    )
    assert "nvm install v20 failed" in text, (
        "build_ui.sh must surface `nvm install v20` failure with an "
        "error message and non-zero exit (LIT-2723)."
    )
    assert "Failed to switch to Node.js v20" in text, (
        "build_ui.sh must surface `nvm use v20` failure with an error "
        "message and non-zero exit."
    )
    assert "node -v" in text, (
        "build_ui.sh must read `node -v` to verify the active Node "
        "version after `nvm use v20`."
    )
    assert "v20." in text, (
        "build_ui.sh must check that `node -v` output starts with "
        "`v20.` and abort otherwise (LIT-2723)."
    )
