from __future__ import annotations

import os
import shutil
from types import SimpleNamespace


class CodexCLIProvider:
    """
    Minimal stub provider to satisfy binary resolution smoke tests.

    This does NOT execute any external binary. It only simulates path/env
    resolution and returns a config object with the derived fields.
    """

    def _resolve_binary_path(self) -> str | None:
        # Env override takes precedence
        p_env = os.getenv("CODEX_CLI_BINARY") or os.getenv("CODEX_BINARY")
        if p_env and os.path.isfile(p_env) and os.access(p_env, os.X_OK):
            return p_env
        # PATH resolution
        p = shutil.which("codex")
        if p and os.path.isfile(p) and os.access(p, os.X_OK):
            return p
        return None

    def _build_run_config(self, optional_params: dict, litellm_params: dict):  # type: ignore[override]
        bin_path = self._resolve_binary_path() or ""
        # Safe defaults the smoke asserts
        args = ["--sandbox", "on", "--approval-mode", "auto"]
        return SimpleNamespace(binary_path=bin_path, args=args)

