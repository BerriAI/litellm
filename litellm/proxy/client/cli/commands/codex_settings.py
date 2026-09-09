"""Codex's config.toml: what `lite configure codex` writes there and how it undoes it.

Codex ignores OPENAI_BASE_URL and routes through a named provider, so the wiring is a
`[model_providers.litellm]` table plus `model_provider` at the root. A Codex launched from the
desktop app never sees the shell environment, which is why the credential is written into the
table (a static key) or read through a command Codex runs itself (the `lite login` credential),
never through an env var.
"""

import hashlib
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Final

from litellm.litellm_core_utils.private_json import commit_staged_json

from .agent_config import (
    ROOT_SECTION,
    ConfigDocument,
    TomlDocument,
    UnconfigureOutcome,
    configure_document,
    unconfigure_document,
)
from .claude_settings import ClaudeCredential, KeepModel, ModelChoice, StartOn, StaticToken

CODEX_HOME_ENV: Final = "CODEX_HOME"
CODEX_PROVIDER_ID: Final = "litellm"
MODEL_PROVIDERS_KEY: Final = "model_providers"
MODEL_PROVIDER_KEY: Final = "model_provider"
MODEL_KEY: Final = "model"
MODEL_CONTEXT_WINDOW_KEY: Final = "model_context_window"
OWNED_ROOT_KEYS: Final = (MODEL_PROVIDER_KEY, MODEL_KEY, MODEL_CONTEXT_WINDOW_KEY)
OWNED_SECTIONS: Final[Mapping[str, Sequence[str]]] = MappingProxyType(
    {ROOT_SECTION: OWNED_ROOT_KEYS, MODEL_PROVIDERS_KEY: (CODEX_PROVIDER_ID,)}
)
PINNED_KEYS: Final = ((ROOT_SECTION, MODEL_KEY), (ROOT_SECTION, MODEL_CONTEXT_WINDOW_KEY))


def codex_home() -> Path:
    """Where Codex keeps its config: ~/.codex unless CODEX_HOME relocates it."""
    override: Final = os.environ.get(CODEX_HOME_ENV)
    return Path(override).expanduser() if override else Path.home() / ".codex"


def codex_config_path(environ: Mapping[str, str]) -> Path:
    home: Final = environ.get(CODEX_HOME_ENV)
    return Path(home).expanduser() / "config.toml" if home else Path.home() / ".codex" / "config.toml"


def codex_configure_state_path(config_path: Path) -> Path:
    default: Final = Path.home() / ".codex" / "config.toml"
    state_root: Final = Path.home() / ".litellm" / "codex_configure_state"
    default_state: Final = state_root.parent / "codex_configure_state.json"
    if config_path.resolve() == default.resolve():
        return default_state
    digest: Final = hashlib.sha256(str(config_path.resolve()).encode()).hexdigest()
    return state_root / f"{digest}.json"


CODEX_CONFIG_PATH: Final = codex_home() / "config.toml"
CODEX_CONFIGURE_STATE_PATH: Final = Path.home() / ".litellm" / "codex_configure_state.json"


def codex_provider(
    base_url: str, credential: ClaudeCredential, print_token: Callable[[], Sequence[str]]
) -> Mapping[str, object]:
    """The `[model_providers.litellm]` table: HTTP/SSE Responses transport, credential inline or by command.

    supports_websockets is off because the proxy does not speak the Responses WebSocket protocol. The
    print-token argv is resolved only for the login credential, so a static key never needs `lite`
    on PATH.
    """
    auth: Final = (
        (("experimental_bearer_token", credential.token),)
        if isinstance(credential, StaticToken)
        else (("auth", _auth_command(print_token())),)
    )
    return MappingProxyType(
        dict(
            (
                ("name", "LiteLLM proxy"),
                ("base_url", base_url.rstrip("/") + "/v1"),
                ("wire_api", "responses"),
                ("supports_websockets", False),
                *auth,
            )
        )
    )


def _auth_command(argv: Sequence[str]) -> Mapping[str, object]:
    return MappingProxyType({"command": argv[0], "args": tuple(argv[1:]), "timeout_ms": 5000})


def apply_codex_merge(
    document: ConfigDocument,
    base_url: str,
    credential: ClaudeCredential,
    print_token: Callable[[], Sequence[str]],
    model: str | None,
    context_window: int | None,
) -> None:
    """Point Codex at the proxy in place; `model` and `context_window` are written only when given.

    Codex sizes its context from its own catalog, which knows nothing about proxy models, so the
    proxy's limit for the chosen model is written as model_context_window alongside the pin. A pin
    with no known window leaves the key to whatever release put there, never a stale one.
    """
    document.set_value(ROOT_SECTION, MODEL_PROVIDER_KEY, CODEX_PROVIDER_ID)
    document.set_value(MODEL_PROVIDERS_KEY, CODEX_PROVIDER_ID, codex_provider(base_url, credential, print_token))
    if model is not None:
        document.set_value(ROOT_SECTION, MODEL_KEY, model)
    if context_window is not None:
        document.set_value(ROOT_SECTION, MODEL_CONTEXT_WINDOW_KEY, context_window)


def configure_codex_config(
    base_url: str,
    credential: ClaudeCredential,
    print_token: Callable[[], Sequence[str]],
    model: ModelChoice,
    context_window: int | None,
    config_path: Path,
    state_path: Path,
    commit: Callable[[str, str], None] = commit_staged_json,
) -> None:
    """Persistently route Codex through base_url, recording how to undo it (see configure_document).

    Both a re-pin and an unpin first let go of the model and window an earlier configure wrote,
    so a re-pin to a model whose window the proxy does not report cannot keep the old model's.
    """
    pinned: Final = model.model if isinstance(model, StartOn) else None
    configure_document(
        config_path,
        state_path,
        (),
        TomlDocument.parse,
        OWNED_SECTIONS,
        lambda document: apply_codex_merge(document, base_url, credential, print_token, pinned, context_window),
        release=() if isinstance(model, KeepModel) else PINNED_KEYS,
        commit=commit,
    )


def unconfigure_codex_config(config_path: Path, state_path: Path) -> UnconfigureOutcome:
    """Undo `lite configure codex`, restoring only the keys the user has not changed since."""
    return unconfigure_document(config_path, state_path, (), TomlDocument.parse, "Codex")


__all__ = (
    "CODEX_CONFIGURE_STATE_PATH",
    "CODEX_CONFIG_PATH",
    "CODEX_HOME_ENV",
    "CODEX_PROVIDER_ID",
    "OWNED_ROOT_KEYS",
    "OWNED_SECTIONS",
    "PINNED_KEYS",
    "apply_codex_merge",
    "codex_config_path",
    "codex_configure_state_path",
    "codex_home",
    "codex_provider",
    "configure_codex_config",
    "unconfigure_codex_config",
)
