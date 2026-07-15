import atexit
import json
import secrets
import signal
import threading
from types import FrameType

import click
import yaml
from pydantic import JsonValue, TypeAdapter

from ..up import CLAUDE_SETTINGS_PATH, load_json_or_empty, restore_claude_settings, write_backup
from ..up import BackupRecord as ClaudeBackupRecord
from .process import (
    AUTOROUTE_DIR,
    CONFIG_PATH,
    LOG_PATH,
    PidRecord,
    ProcessLaunchError,
    allocate_free_port,
    clear_pid_record,
    is_running,
    launch_proxy,
    poll_liveliness,
    read_pid_record,
    secure_create,
    stream_log,
    terminate,
    write_pid_record,
)
from .settings import merge_claude_settings_static_token
from .wizard import run_configure_wizard

AUTOROUTE_BACKUP_PATH = AUTOROUTE_DIR / "claude_settings_backup.json"

_GENERATED_CONFIG_ADAPTER = TypeAdapter(dict[str, JsonValue])


def _mint_and_embed_master_key() -> str:
    """Generate a fresh key for this session and write it into the generated config.yaml.

    Must go under general_settings, not litellm_settings -- the proxy server only ever
    reads general_settings.master_key (proxy_server.py:4530) to authenticate requests. A
    key placed under litellm_settings is silently ignored, leaving the ephemeral proxy with
    no real auth: any request reaches it regardless of the token Claude Code sends.
    """
    master_key = secrets.token_urlsafe(32)
    with open(CONFIG_PATH, "r") as f:
        generated = _GENERATED_CONFIG_ADAPTER.validate_python(yaml.safe_load(f))
    general_settings = generated.get("general_settings")
    updated_settings: dict[str, JsonValue] = {
        **(general_settings if isinstance(general_settings, dict) else {}),
        "master_key": master_key,
    }
    updated: dict[str, JsonValue] = {**generated, "general_settings": updated_settings}
    with secure_create(CONFIG_PATH) as f:
        yaml.safe_dump(updated, f, sort_keys=False)
    return master_key


@click.group(name="autoroute")
def autoroute_group() -> None:
    """QA complexity-based auto-routing against models your key can already use"""


@autoroute_group.command("configure")
@click.pass_context
def configure(ctx: click.Context) -> None:
    """Discover accessible models and generate an ephemeral auto-router config"""
    run_configure_wizard(ctx)


@autoroute_group.command("up")
def up() -> None:
    """Launch the ephemeral auto-router proxy and route Claude Code through it"""
    if not CONFIG_PATH.exists():
        raise click.ClickException("No config found. Run `lite autoroute configure` first.")

    existing_pid = read_pid_record()
    if existing_pid is not None and is_running(existing_pid.pid):
        raise click.ClickException(
            "An ephemeral proxy is already running (lite autoroute up looks already active). "
            "Run `lite autoroute down` first."
        )

    if AUTOROUTE_BACKUP_PATH.exists():
        raise click.ClickException(
            f"{AUTOROUTE_BACKUP_PATH} already exists -- `lite autoroute up` looks like it's already "
            "running (or crashed without cleanup). Run `lite autoroute down` first."
        )

    master_key = _mint_and_embed_master_key()
    port = allocate_free_port()
    base_url = f"http://127.0.0.1:{port}"
    process = launch_proxy(CONFIG_PATH, port, LOG_PATH)
    write_pid_record(PidRecord(pid=process.pid, port=port, config_path=str(CONFIG_PATH), log_path=str(LOG_PATH)))

    try:
        poll_liveliness(base_url, LOG_PATH, process)
    except ProcessLaunchError as e:
        terminate(process.pid)
        clear_pid_record()
        raise click.ClickException(str(e))

    original_existed = CLAUDE_SETTINGS_PATH.exists()
    original_settings = load_json_or_empty(CLAUDE_SETTINGS_PATH)
    write_backup(
        ClaudeBackupRecord(existed=original_existed, content=original_settings if original_existed else None),
        AUTOROUTE_BACKUP_PATH,
    )
    merged = merge_claude_settings_static_token(original_settings, base_url, master_key)
    CLAUDE_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with secure_create(CLAUDE_SETTINGS_PATH) as f:
        json.dump(merged, f, indent=2)

    click.echo(f"litellm: ephemeral auto-router proxy up at {base_url} (pid {process.pid})")
    click.echo("Claude Code sessions started now will route through it. Press Ctrl-C to stop and restore.")

    stop_event = threading.Event()
    restored = threading.Lock()

    def _teardown() -> None:
        if not restored.acquire(blocking=False):
            return
        terminate(process.pid)
        clear_pid_record()
        restore_claude_settings(CLAUDE_SETTINGS_PATH, AUTOROUTE_BACKUP_PATH)
        click.echo("\nStopped ephemeral proxy and restored Claude Code settings.")
        click.echo(
            f"Restart any Claude Code session still open from this session, or another local account could "
            f"bind the now-free port {port} and receive its requests. Do not use `lite autoroute up` on a "
            f"shared or multi-tenant host."
        )

    def _handle_signal(_signum: int, _frame: FrameType | None) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    atexit.register(_teardown)

    log_thread = threading.Thread(target=stream_log, args=(LOG_PATH, stop_event), daemon=True)
    log_thread.start()

    stop_event.wait()
    _teardown()


@autoroute_group.command("down")
def down() -> None:
    """Restore Claude Code settings and stop a leftover ephemeral proxy, if any"""
    record: PidRecord | None = read_pid_record()
    if record is not None and is_running(record.pid):
        terminate(record.pid)
        click.echo(f"Stopped leftover ephemeral proxy (pid {record.pid}).")
    clear_pid_record()

    restored = restore_claude_settings(CLAUDE_SETTINGS_PATH, AUTOROUTE_BACKUP_PATH)
    if restored is None:
        click.echo("Nothing to restore.")
    elif restored.existed:
        click.echo(f"Restored {CLAUDE_SETTINGS_PATH} to its original contents.")
    else:
        click.echo(f"Removed {CLAUDE_SETTINGS_PATH} (it did not exist before `lite autoroute up`).")


__all__ = ["autoroute_group"]
