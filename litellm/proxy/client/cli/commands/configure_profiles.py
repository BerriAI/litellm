"""Reusable agent setup, separate from the settings writers' undo receipts."""

import hashlib
import os
from collections.abc import Generator, Sequence
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Final, Literal, TypeAlias

import click
from filelock import FileLock, Timeout
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from litellm.litellm_core_utils.private_json import (
    commit_staged_json,
    discard_staged_json,
    ensure_private_dir,
    stage_private_json,
)

from .agents import codex_config_path
from .claude_settings import claude_settings_path, configure_state_path
from .codex_settings import codex_configure_state_path
from .config import normalize_base_url

Target: TypeAlias = Literal["claude", "codex"]
TARGETS: Final[tuple[Target, ...]] = ("claude", "codex")


class SavedSetup(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    version: Literal[1] = 1
    target: Target
    settings_path: str
    base_url: str
    api_key: str = Field(repr=False)
    model: str | None

    @field_validator("base_url")
    @classmethod
    def normalized_gateway(cls, value: str) -> str:
        try:
            if normalize_base_url(value).removesuffix("/v1") != value:
                raise ValueError("Gateway must be normalized")
        except click.UsageError as error:
            raise ValueError("Invalid gateway URL") from error
        return value

    @field_validator("api_key")
    @classmethod
    def valid_key(cls, value: str) -> str:
        if not value or any(ord(char) <= 32 or ord(char) == 127 for char in value):
            raise ValueError("Invalid virtual key")
        return value

    @field_validator("model")
    @classmethod
    def valid_model(cls, value: str | None) -> str | None:
        if value is not None and (not value or any(ord(char) < 32 or ord(char) == 127 for char in value)):
            raise ValueError("Invalid model choice")
        return value


def settings_path_for(target: Target) -> Path:
    return claude_settings_path(os.environ) if target == "claude" else codex_config_path(os.environ)


def receipt_path_for(target: Target, settings_path: Path) -> Path:
    return configure_state_path(settings_path) if target == "claude" else codex_configure_state_path(settings_path)


def setup_profile_path(target: Target, settings_path: Path) -> Path:
    receipt: Final = receipt_path_for(target, settings_path)
    return receipt.with_name(f"{receipt.stem}_profile.json")


def read_saved_setup(target: Target) -> SavedSetup | None:
    settings_path: Final = settings_path_for(target)
    path: Final = setup_profile_path(target, settings_path)
    try:
        payload: Final = path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as error:
        raise click.ClickException(
            f"Could not read saved {target} setup at {path}; no settings were changed"
        ) from error
    try:
        saved: Final = SavedSetup.model_validate_json(payload)
        if (
            saved.target != target
            or saved.settings_path != str(settings_path.resolve())
            or (target == "codex" and saved.model is None)
        ):
            raise ValueError("Invalid saved setup")
        return saved
    except (ValidationError, ValueError, click.UsageError) as error:
        raise click.ClickException(
            f"Saved {target} setup at {path} is invalid or unsupported. "
            f"Run `lite unconfigure {target} --forget` to discard it; no settings were changed"
        ) from error


def _lock_path(target: Target) -> Path:
    digest: Final = hashlib.sha256(f"{target}:{settings_path_for(target).resolve()}".encode()).hexdigest()
    return Path.home() / ".litellm" / "setup-locks" / f"{digest}.lock"


@contextmanager
def setup_locks(targets: Sequence[Target]) -> Generator[None, None, None]:
    with ExitStack() as stack:
        try:
            for path in tuple(_lock_path(target) for target in sorted(frozenset(targets))):
                ensure_private_dir(path.parent)
                stack.enter_context(FileLock(str(path), timeout=10, mode=0o600))
        except (OSError, Timeout) as error:
            raise click.ClickException(
                "Could not lock agent setup; retry when other configure commands finish"
            ) from error
        yield


def save_setup(saved: SavedSetup) -> None:
    path: Final = setup_profile_path(saved.target, settings_path_for(saved.target))
    try:
        ensure_private_dir(path.parent)
        staged: Final = stage_private_json(
            str(path),
            {
                "version": saved.version,
                "target": saved.target,
                "settings_path": saved.settings_path,
                "base_url": saved.base_url,
                "api_key": saved.api_key,
                "model": saved.model,
            },
        )
    except OSError as error:
        raise click.ClickException(
            f"Could not save {saved.target} setup; no {saved.target} settings were changed"
        ) from error
    try:
        commit_staged_json(staged, str(path))
    except OSError as error:
        raise click.ClickException(
            f"Could not save {saved.target} setup; no {saved.target} settings were changed"
        ) from error
    finally:
        discard_staged_json(staged)


def forget_saved_setup(target: Target) -> None:
    path: Final = setup_profile_path(target, settings_path_for(target))
    try:
        path.unlink(missing_ok=True)
    except OSError as error:
        raise click.ClickException(f"Could not remove saved {target} setup at {path}") from error
