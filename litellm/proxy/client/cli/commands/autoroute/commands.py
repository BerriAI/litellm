import atexit
import json
import secrets
import signal
import threading
from types import FrameType

import click
import yaml
from pydantic import JsonValue, TypeAdapter, ValidationError

from ..up import CLAUDE_SETTINGS_PATH, UpError, load_json_or_empty, restore_claude_settings, write_backup
from ..up import BackupRecord as ClaudeBackupRecord
from .config import master_key_from_config
from .process import (
    AUTOROUTE_DIR,
    CONFIG_PATH,
    DEFAULT_AUTOROUTE_PORT,
    LOG_PATH,
    PidRecord,
    ProcessLaunchError,
    clear_pid_record,
    is_port_available,
    is_running,
    launch_proxy,
    missing_proxy_runtime_modules,
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


def _ensure_master_key() -> str:
    """Reuse the master key already persisted in the generated config.yaml, minting one only when absent.

    The generated config is the single home of the key: the proxy server authenticates against
    general_settings.master_key only (a key under litellm_settings is silently ignored, which
    would leave the ephemeral proxy with no real auth), and the file is written 0600 via
    secure_create. Reusing that persisted value keeps the key stable across `up` runs, so a
    client configured against one session keeps working in the next.
    """
    with open(CONFIG_PATH, "r") as f:
        try:
            generated = _GENERATED_CONFIG_ADAPTER.validate_python(yaml.safe_load(f))
        except (yaml.YAMLError, ValidationError):
            raise click.ClickException(
                f"{CONFIG_PATH} is empty or corrupt. Run `lite autoroute configure` again to regenerate it."
            )
    persisted = master_key_from_config(generated)
    if persisted is not None:
        return persisted
    master_key = secrets.token_urlsafe(32)
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
@click.option(
    "--port",
    type=click.IntRange(1, 65535),
    default=DEFAULT_AUTOROUTE_PORT,
    show_default=True,
    help="Loopback port for the ephemeral proxy; stable across runs so configured clients keep working.",
)
def up(port: int) -> None:
    """Launch the ephemeral auto-router proxy and route Claude Code through it"""
    if not CONFIG_PATH.exists():
        raise click.ClickException("No config found. Run `lite autoroute configure` first.")

    missing = missing_proxy_runtime_modules()
    if missing:
        raise click.ClickException(
            "lite autoroute up launches a local litellm proxy, which needs the proxy runtime that the "
            f"thin `litellm[cli]` install does not include (missing: {', '.join(missing)}). Install the "
            "proxy runtime with `uv tool install --force 'litellm[proxy]'`, or to QA a branch, "
            "`curl -fsSL https://raw.githubusercontent.com/BerriAI/litellm/<branch>/scripts/install.sh | "
            "LITELLM_CLI_REF=<branch> sh`."
        )

    try:
        existing_pid = read_pid_record()
    except UpError as e:
        raise click.ClickException(str(e))
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

    if port == 4000:
        raise click.ClickException(
            "Port 4000 is the litellm proxy's own default and its launcher silently rebinds it to a random "
            "port when busy; pick a different --port."
        )

    if not is_port_available(port):
        raise click.ClickException(
            f"Port {port} on 127.0.0.1 is already in use. If a previous `lite autoroute up` is still "
            "running or crashed, run `lite autoroute down`; otherwise pick a different port with --port."
        )

    master_key = _ensure_master_key()
    base_url = f"http://127.0.0.1:{port}"
    process = launch_proxy(CONFIG_PATH, port, LOG_PATH)
    write_pid_record(PidRecord(pid=process.pid, port=port, config_path=str(CONFIG_PATH), log_path=str(LOG_PATH)))

    try:
        poll_liveliness(base_url, LOG_PATH, process)
    except ProcessLaunchError as e:
        terminate(process.pid)
        clear_pid_record()
        raise click.ClickException(str(e))

    try:
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
    except UpError as e:
        terminate(process.pid)
        clear_pid_record()
        raise click.ClickException(str(e))

    click.echo(f"litellm: ephemeral auto-router proxy up at {base_url} (pid {process.pid})")
    click.echo("Claude Code sessions started now will route through it. Press Ctrl-C to stop and restore.")

    stop_event = threading.Event()
    restored = threading.Lock()

    def _teardown() -> None:
        if not restored.acquire(blocking=False):
            return
        terminate(process.pid)
        clear_pid_record()
        try:
            restore_claude_settings(CLAUDE_SETTINGS_PATH, AUTOROUTE_BACKUP_PATH)
        except UpError as e:
            # Runs from atexit/a signal handler too, outside Click's own exception
            # handling -- raising here would only produce an unhandled-exception
            # warning on stderr, not a clean message.
            click.echo(str(e), err=True)
            return
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
    try:
        record: PidRecord | None = read_pid_record()
    except UpError as e:
        # down is the crash-recovery path -- a corrupt pid record must not block it; clear the
        # unusable record and keep going rather than leaving the user with no way to clean up.
        click.echo(f"{e} Clearing it and continuing cleanup.", err=True)
        record = None
    if record is not None and is_running(record.pid):
        terminate(record.pid)
        click.echo(f"Stopped leftover ephemeral proxy (pid {record.pid}).")
    clear_pid_record()

    try:
        restored = restore_claude_settings(CLAUDE_SETTINGS_PATH, AUTOROUTE_BACKUP_PATH)
    except UpError as e:
        raise click.ClickException(str(e))
    if restored is None:
        click.echo("Nothing to restore.")
    elif restored.existed:
        click.echo(f"Restored {CLAUDE_SETTINGS_PATH} to its original contents.")
    else:
        click.echo(f"Removed {CLAUDE_SETTINGS_PATH} (it did not exist before `lite autoroute up`).")


__all__ = ["autoroute_group"]
