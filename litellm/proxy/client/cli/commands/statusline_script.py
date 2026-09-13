"""Claude Code status line and Codex Stop hook for auto-routed sessions.

`lite` copies this file verbatim to ~/.litellm/statusline.py and registers it as Claude
Code's `statusLine` command and as Codex's `[[hooks.Stop]]` command, so it must stay
standard-library only and must never import litellm. Claude Code re-runs it on every
status refresh (about every 300ms while typing), so the proxy is asked at most once per
TTL per session and every other refresh is served from a small on-disk cache that holds
only the proxy's answer, never the key.

Claude Code pipes a JSON payload on stdin (session_id, transcript_path, model); the routed
model is the `message.model` of the latest foreground assistant line in the transcript,
which is the proxy's response `model` field. That only names the tier model when the
auto-router deployment sets `return_raw_model_name: true`; otherwise it is the alias the
client requested. Codex pipes its Stop event instead (hook_event_name, session_id) and has
no transcript to read, so the routed model comes from the proxy's session record and the
result is printed as a `systemMessage` for the transcript. The proxy key is read from the
agent's own environment (the static token `lite configure claude` writes); nothing here
spawns a credential helper.

Cost figures come from GET /auto_router/session on the proxy, which reads the per-session
rollup written by the spend flush. That flush is asynchronous, so a turn's cost lands a
second or two after the turn; the cache TTL absorbs it.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import IO, Final, NamedTuple, Protocol
from urllib.parse import urlencode

SESSION_ENDPOINT: Final = "/auto_router/session"
CACHE_TTL_SECONDS: Final = 5.0
FETCH_TIMEOUT_SECONDS: Final = 3
BAR_WIDTH: Final = 24
BAR_FULL: Final = "\u2588"
BAR_EMPTY: Final = "\u2591"
SEPARATOR: Final = " \u00b7 "
TRANSCRIPT_SCAN_LIMIT_BYTES: Final = 4 * 1024 * 1024
CLAUDE_BASE_URL_ENV_KEYS: Final = ("ANTHROPIC_BASE_URL",)
CLAUDE_API_KEY_ENV_KEYS: Final = ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY")
CODEX_BASE_URL_ENV_KEYS: Final = ("OPENAI_BASE_URL",)
CODEX_API_KEY_ENV_KEYS: Final = ("OPENAI_API_KEY",)
CODEX_STOP_EVENT: Final = "Stop"
SYNTHETIC_MODEL: Final = "<synthetic>"
LITELLM_LABEL: Final = "LiteLLM"
RESET: Final = "\033[0m"
BOLD: Final = "\033[1m"
DIM: Final = "\033[90m"
LITELLM_COLOR: Final = "\033[38;2;79;70;229m"
BASELINE_COLOR: Final = "\033[38;2;217;119;87m"
EMPTY: Final[Mapping[str, object]] = MappingProxyType({})
EMPTY_ENV: Final[Mapping[str, str]] = MappingProxyType({})


class Session(NamedTuple):
    router_name: str
    last_model: str
    spend: float
    baseline_spend: float
    baseline_model: str | None


class Credentials(NamedTuple):
    base_url: str
    api_key: str

    @property
    def usable(self) -> bool:
        return bool(self.base_url and self.api_key)


class Fetched(NamedTuple):
    session: Session | None
    definitive: bool


class Fetch(Protocol):
    def __call__(self, credentials: Credentials, session_id: str) -> Fetched: ...


def as_mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, dict) else EMPTY


def as_str(value: object) -> str:
    return value if isinstance(value, str) else ""


def printable(value: object) -> str:
    """Labels come from the transcript, the proxy, and Claude Code's model cache, none of which this script
    controls, and every one is written to a terminal: a control character (ESC, BEL, C1) in a model name
    could redraw the screen or set the clipboard, so only printable text survives."""
    return "".join(character for character in as_str(value) if character.isprintable())


def load_json(raw: bytes | str) -> object:
    try:
        return json.loads(raw)
    except ValueError:
        return None


def resolve_base_url(env: Mapping[str, str], keys: tuple[str, ...]) -> str:
    raw: Final = next((env[key] for key in keys if env.get(key)), "").strip().rstrip("/")
    return raw.removesuffix("/v1")


def resolve_api_key(env: Mapping[str, str], keys: tuple[str, ...]) -> str:
    return next((env[key] for key in keys if env.get(key)), "").strip()


def claude_credentials(env: Mapping[str, str]) -> Credentials:
    """Claude Code's own resolution order, so the key-scoped lookup runs as the principal that wrote the rows:
    ANTHROPIC_AUTH_TOKEN, then ANTHROPIC_API_KEY. A `lite` variable such as LITELLM_PROXY_API_KEY is not a
    key Claude Code ever sends, so honoring it would ask as someone else. An apiKeyHelper is never run: a
    status line refreshes every few hundred milliseconds, and spawning a credential helper that often is
    how a keychain prompt ends up on screen a hundred times."""
    return Credentials(resolve_base_url(env, CLAUDE_BASE_URL_ENV_KEYS), resolve_api_key(env, CLAUDE_API_KEY_ENV_KEYS))


def codex_credentials(env: Mapping[str, str]) -> Credentials:
    return Credentials(resolve_base_url(env, CODEX_BASE_URL_ENV_KEYS), resolve_api_key(env, CODEX_API_KEY_ENV_KEYS))


def _transcript_line_model(line: bytes) -> str:
    """A `<synthetic>` model is Claude Code's own marker for a locally produced message (an API error, a
    resume note), not a served model, so it is skipped like a sidechain line."""
    item: Final = as_mapping(load_json(line))
    if item.get("type") != "assistant" or item.get("isSidechain") is True or item.get("agentId"):
        return ""
    model: Final = printable(as_mapping(item.get("message")).get("model"))
    return "" if model == SYNTHETIC_MODEL else model


def latest_transcript_model(transcript_path: str) -> str:
    if not transcript_path:
        return ""
    try:
        with Path(transcript_path).open("rb") as transcript:
            size: Final = transcript.seek(0, os.SEEK_END)
            transcript.seek(max(0, size - TRANSCRIPT_SCAN_LIMIT_BYTES))
            tail: Final = transcript.read()
    except OSError:
        return ""
    return next((model for line in reversed(tail.split(b"\n")) if (model := _transcript_line_model(line))), "")


def model_label(model: str, config_dir: Path) -> str:
    bare: Final = model.rsplit("/", 1)[-1]
    try:
        raw: Final = (config_dir / "cache" / "gateway-models.json").read_bytes()
    except OSError:
        return bare
    listed: Final = as_mapping(load_json(raw)).get("models")
    if not isinstance(listed, list):
        return bare
    entries: Final = tuple(as_mapping(entry) for entry in listed)
    return next(
        (
            printable(entry.get("display_name"))
            for entry in entries
            if entry.get("id") in (model, bare) and printable(entry.get("display_name"))
        ),
        bare,
    )


def baseline_label(model: str, config_dir: Path) -> str:
    labelled: Final = model_label(model, config_dir)
    if labelled != model.rsplit("/", 1)[-1]:
        return labelled
    return " ".join(word.capitalize() for word in labelled.replace("-", " ").split())


def fetch_session(credentials: Credentials, session_id: str) -> Fetched:
    """Any 4xx is this credential's definite answer (no row, no access, expired login) and is cached for the
    TTL; a 5xx or transport failure is not, so the next refresh tries again."""
    query: Final = urlencode((("session_id", session_id),))
    request: Final = urllib.request.Request(
        f"{credentials.base_url}{SESSION_ENDPOINT}?{query}",
        headers={  # mutable-ok: urllib.request.Request takes a dict
            "Authorization": f"Bearer {credentials.api_key}",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=FETCH_TIMEOUT_SECONDS) as response:
            raw: Final[bytes] = response.read()
    except urllib.error.HTTPError as error:
        return Fetched(session=None, definitive=400 <= error.code < 500)
    except (urllib.error.URLError, OSError):
        return Fetched(session=None, definitive=False)
    session: Final = _session_from_payload(as_mapping(load_json(raw)))
    return Fetched(session=session, definitive=session is not None)


def _session_from_payload(payload: Mapping[str, object]) -> Session | None:
    router_name: Final = printable(payload.get("router_name"))
    last_model: Final = printable(payload.get("last_model"))
    spend: Final = payload.get("spend")
    baseline_spend: Final = payload.get("baseline_spend")
    if not router_name or not last_model:
        return None
    if not isinstance(spend, (int, float)) or not isinstance(baseline_spend, (int, float)):
        return None
    return Session(
        router_name=router_name,
        last_model=last_model,
        spend=float(spend),
        baseline_spend=float(baseline_spend),
        baseline_model=printable(payload.get("baseline_model")) or None,
    )


def cache_path(cache_dir: Path, credentials: Credentials, session_id: str) -> Path:
    identity: Final = "\n".join((credentials.base_url, credentials.api_key, session_id))
    return cache_dir / hashlib.sha256(identity.encode()).hexdigest()


def load_session(
    credentials: Credentials,
    session_id: str,
    cache_dir: Path,
    fetch: Fetch = fetch_session,
    now: Callable[[], float] = time.time,
) -> Session | None:
    path: Final = cache_path(cache_dir, credentials, session_id)
    cached: Final = _read_cache(path)
    fetched_at: Final = cached.get("fetched_at")
    if isinstance(fetched_at, (int, float)) and now() - fetched_at < CACHE_TTL_SECONDS:
        return _session_from_payload(as_mapping(cached.get("session")))
    fetched: Final = fetch(credentials, session_id)
    if fetched.definitive:
        _write_cache(path, fetched.session, now())
    return fetched.session


NOFOLLOW: Final = getattr(os, "O_NOFOLLOW", 0)


def cache_dir_name() -> str:
    return f"litellm-statusline-{os.getuid()}" if hasattr(os, "getuid") else "litellm-statusline"


def _own_private_dir(directory: Path) -> bool:
    """A shared temp root lets another local user pre-create the directory, so it must be ours and private
    before anything is read or written under it. Windows has no uids or POSIX mode bits and a per-user temp
    directory already, so there it only has to exist and not be a link."""
    try:
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        status: Final = directory.lstat()
    except OSError:
        return False
    if not os.path.isdir(directory) or os.path.islink(directory):
        return False
    if not hasattr(os, "getuid"):
        return True
    return status.st_uid == os.getuid() and not status.st_mode & 0o077


def _read_cache(path: Path) -> Mapping[str, object]:
    if not _own_private_dir(path.parent):
        return EMPTY
    try:
        descriptor: Final = os.open(path, os.O_RDONLY | NOFOLLOW)
        with os.fdopen(descriptor, "rb") as handle:
            return as_mapping(load_json(handle.read()))
    except OSError:
        return EMPTY


def _write_cache(path: Path, session: Session | None, fetched_at: float) -> None:
    """Staged beside the entry and renamed into place, so a refresh reading the entry never sees a torn write."""
    entry: Final = session._asdict() if session else None
    body: Final = json.dumps({"fetched_at": fetched_at, "session": entry})  # mutable-ok: json.dumps takes a dict
    if not _own_private_dir(path.parent):
        return
    try:
        descriptor, staged = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    except OSError:
        return
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(body)
        os.replace(staged, path)
    except OSError:
        Path(staged).unlink(missing_ok=True)


def _bar(fraction: float, color: str, width: int, use_color: bool) -> str:
    filled: Final = round(max(0.0, min(1.0, fraction)) * width)
    if not use_color:
        return BAR_FULL * filled + BAR_EMPTY * (width - filled)
    return f"{color}{BAR_FULL * filled}{DIM}{BAR_EMPTY * (width - filled)}{RESET}"


def render(model: str, session: Session | None, config_dir: Path, use_color: bool, bar_width: int = BAR_WIDTH) -> str:
    def paint(code: str, text: str) -> str:
        return f"{code}{text}{RESET}" if use_color else text

    routed: Final = paint(BOLD, f"Routed to: {model}")
    if session is None:
        return routed
    header: Final = f"{session.router_name}{SEPARATOR}{routed}"
    if session.baseline_model is None or session.baseline_spend <= 0:
        return header
    reference: Final = baseline_label(session.baseline_model, config_dir)
    pct: Final = (session.baseline_spend - session.spend) / session.baseline_spend * 100
    delta: Final = paint(LITELLM_COLOR, f"{'-' if pct >= 0 else '+'}{abs(round(pct))}% vs {reference}")
    peak: Final = max(session.spend, session.baseline_spend)
    label_width: Final = max(len(LITELLM_LABEL), len(reference))
    rows: Final = (
        (LITELLM_LABEL, session.spend, LITELLM_COLOR),
        (reference, session.baseline_spend, BASELINE_COLOR),
    )
    lines: Final = (
        f"{paint(DIM, label.ljust(label_width))} {_bar(amount / peak, color, bar_width, use_color)} "
        f"{paint(DIM, f'${amount:.2f}')}"
        for label, amount, color in rows
    )
    return "\n".join((f"{header}  {delta}", *lines))


def color_enabled(env: Mapping[str, str]) -> bool:
    return env.get("NO_COLOR") is None and env.get("TERM", "") not in ("", "dumb")


def status_line(
    payload: Mapping[str, object], env: Mapping[str, str], config_dir: Path, cache_dir: Path, fetch: Fetch
) -> str:
    fallback: Final = printable(as_mapping(payload.get("model")).get("display_name"))
    served: Final = latest_transcript_model(as_str(payload.get("transcript_path")))
    if not served:
        return fallback or "claude"
    label: Final = model_label(served, config_dir)
    session_id: Final = as_str(payload.get("session_id"))
    credentials: Final = claude_credentials(env)
    if not session_id or not credentials.usable:
        return render(label, None, config_dir, color_enabled(env))
    session: Final = load_session(credentials, session_id, cache_dir, fetch)
    return render(label, session, config_dir, color_enabled(env))


def codex_stop_message(
    payload: Mapping[str, object], env: Mapping[str, str], config_dir: Path, cache_dir: Path, fetch: Fetch
) -> str:
    """No cache here: the Stop hook runs once per turn, and a first turn's cached absence would hide the record
    the next turn finds."""
    session_id: Final = as_str(payload.get("session_id"))
    credentials: Final = codex_credentials(env)
    if not session_id or not credentials.usable:
        return ""
    session: Final = fetch(credentials, session_id).session
    if session is None:
        return ""
    text: Final = render(model_label(session.last_model, config_dir), session, config_dir, use_color=False)
    return json.dumps({"systemMessage": f"\n{text}"})  # mutable-ok: json.dumps takes a dict


def run(stdin: IO[str], stdout: IO[str], env: Mapping[str, str], fetch: Fetch = fetch_session) -> None:
    """A failure renders each mode's own quiet fallback: Claude Code gets the label it already knows, Codex gets
    nothing at all rather than a bare string it would reject as hook JSON."""
    body: Final = as_mapping(load_json(stdin.read()))
    codex: Final = body.get("hook_event_name") == CODEX_STOP_EVENT
    config_dir: Final = Path(env.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude")
    cache_dir: Final = (
        Path(env.get("TMPDIR") or env.get("TEMP") or env.get("TMP") or tempfile.gettempdir()) / cache_dir_name()
    )
    try:
        text: Final = (
            codex_stop_message(body, env, config_dir, cache_dir, fetch)
            if codex
            else status_line(body, env, config_dir, cache_dir, fetch)
        )
    except Exception:  # noqa: BLE001  # a status line must never break the agent session
        stdout.write("" if codex else printable(as_mapping(body.get("model")).get("display_name")) or "claude")
        return
    stdout.write(text)


if __name__ == "__main__":
    run(sys.stdin, sys.stdout, os.environ)
