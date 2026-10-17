import os
import subprocess
import sys
import webbrowser
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from http import HTTPStatus
from pathlib import Path
from typing import Final, Literal, Protocol

import click
import requests
from filelock import FileLock, Timeout
from typing_extensions import NotRequired, ReadOnly, TypedDict

from litellm.litellm_core_utils.cli_token_utils import get_cli_token_file_path, load_cli_token

from ...exceptions import UnauthorizedError
from .auth import context_secret_vault, get_stored_api_key, replace_stored_token, storage_notice
from .pkce_login import LOGIN_TIMEOUT_SECONDS, PkceCredential, PkceFailure, pkce_token_record, run_pkce_login


class BrowserSignIn(Protocol):
    def unavailable_because(self) -> PkceFailure | None: ...

    def __call__(self, base_url: str, echo: Callable[[str], None]) -> PkceCredential | PkceFailure: ...


class SignInLock(Protocol):
    def acquire(self, timeout: float) -> object: ...

    def release(self) -> object: ...


_TEXT_MODE_BROWSERS: Final = frozenset({"www-browser", "links", "elinks", "lynx", "w3m"})
_SSH_SESSION_VARIABLES: Final = ("SSH_CONNECTION", "SSH_TTY")
_DISPLAY_VARIABLES: Final = ("DISPLAY", "WAYLAND_DISPLAY")
_SIGNED_IN_LOGIN_WITHOUT_KEY_ROW: Final = HTTPStatus.NOT_FOUND
_WAIT_FOR_ANOTHER_SIGN_IN_SECONDS: Final = LOGIN_TIMEOUT_SECONDS + 60


def _runs_in_ci(environ: Mapping[str, str]) -> bool:
    return environ.get("CI", "").strip().lower() not in {"", "0", "false"}


def _ssh_session_without_a_display(environ: Mapping[str, str]) -> bool:
    over_ssh: Final = any(environ.get(name) for name in _SSH_SESSION_VARIABLES)
    return over_ssh and not any(environ.get(name) for name in _DISPLAY_VARIABLES)


_OPEN_URL_IN_BROWSER: Final = "import sys, webbrowser; sys.exit(0 if webbrowser.get().open(sys.argv[1]) else 1)"


def open_browser_detached(url: str) -> PkceFailure | None:
    launcher: Final = subprocess.run(
        (sys.executable, "-I", "-c", _OPEN_URL_IN_BROWSER, url),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return None if launcher.returncode == 0 else PkceFailure("the web browser did not open")


def _live_environ() -> Mapping[str, str]:
    return os.environ


@dataclass(frozen=True, slots=True)
class PkceBrowserSignIn:
    find_browser: Callable[[], webbrowser.BaseBrowser] = webbrowser.get
    environ: Mapping[str, str] = field(default_factory=_live_environ)
    open_browser: Callable[[str], object] = open_browser_detached

    def unavailable_because(self) -> PkceFailure | None:
        if _runs_in_ci(self.environ):
            return PkceFailure("the CI environment variable is set, so lite does not open a browser")
        if "BROWSER" not in self.environ and _ssh_session_without_a_display(self.environ):
            return PkceFailure(
                "this is an SSH session without a display, so a browser would open where you cannot see it"
            )
        try:
            browser: Final = self.find_browser()
        except webbrowser.Error:
            return PkceFailure("no web browser is available on this machine")
        if Path(browser.name).name in _TEXT_MODE_BROWSERS:
            return PkceFailure("the web browser is a text-mode one, which lite cannot show while a command waits")
        return None

    def __call__(self, base_url: str, echo: Callable[[str], None]) -> PkceCredential | PkceFailure:
        echo(
            f"lite is opening your browser to sign in and waits up to {LOGIN_TIMEOUT_SECONDS // 60} minutes "
            "for you to approve it. "
            "A shell or agent that stops commands sooner should run this one with a longer timeout."
        )
        return run_pkce_login(base_url, requests.Session(), open_browser=self.open_browser, echo=echo)


class _SignInContext(TypedDict):
    base_url: ReadOnly[str]
    api_key: ReadOnly[NotRequired[str | None]]
    api_key_from_token_file: ReadOnly[NotRequired[bool]]
    browser_sign_in: ReadOnly[NotRequired[BrowserSignIn]]
    sign_in_lock: ReadOnly[NotRequired[Callable[[], SignInLock]]]


class _SignInContextView(Protocol):
    @property
    def obj(self) -> _SignInContext | None: ...


class _ApiKeySlot(Protocol):
    def __setitem__(self, key: Literal["api_key"], value: str, /) -> None: ...


class _ApiKeySlotView(Protocol):
    @property
    def obj(self) -> _ApiKeySlot: ...


def _sign_in_context(view: _SignInContextView) -> _SignInContext | None:
    return view.obj


def _api_key_slot(view: _ApiKeySlotView) -> _ApiKeySlot:
    return view.obj


@dataclass(frozen=True, slots=True)
class _Completed:
    value: object


@dataclass(frozen=True, slots=True)
class _Unauthorized:
    error: UnauthorizedError


def _attempt(run: Callable[[click.Context], object], ctx: click.Context) -> _Completed | _Unauthorized:
    try:
        return _Completed(run(ctx))
    except UnauthorizedError as error:
        return _Unauthorized(error)


def _attempt_with_key(
    run: Callable[[click.Context], object], ctx: click.Context, api_key: str
) -> _Completed | _Unauthorized:
    slot: Final = _api_key_slot(ctx)
    slot["api_key"] = api_key
    return _attempt(run, ctx)


def _echo_to_stderr(message: str) -> None:
    click.echo(message, err=True)


def _attempt_with_newer_stored_key(
    run: Callable[[click.Context], object], ctx: click.Context, base_url: str, rejected_key: str | None
) -> _Completed | _Unauthorized | None:
    stored_key: Final = get_stored_api_key(expected_base_url=base_url, vault=context_secret_vault(ctx))
    if stored_key is None or stored_key == rejected_key:
        return None
    return _attempt_with_key(run, ctx, stored_key)


def _sign_in_failure(base_url: str, failure: PkceFailure) -> str:
    if failure.proxy_predates_pkce:
        return (
            f"The proxy at {base_url} has no browser sign-in for lite, so it is likely older than v1.99.0. "
            "Upgrade it, run `lite login` if it uses SSO, or set LITELLM_PROXY_API_KEY."
        )
    return (
        f"Could not sign in to {base_url} through the browser: {failure.reason}. "
        "Run `lite login --pkce` from a terminal on a machine with a browser, `lite login` on one without, "
        "or set LITELLM_PROXY_API_KEY."
    )


def _sign_in(ctx: click.Context, base_url: str, sign_in: BrowserSignIn) -> str:
    _echo_to_stderr(f"The proxy at {base_url} needs a sign-in.")
    credential: Final = sign_in(base_url, _echo_to_stderr)
    if isinstance(credential, PkceFailure):
        raise click.ClickException(_sign_in_failure(base_url, credential))
    stored: Final = replace_stored_token(
        pkce_token_record(base_url, credential), requests.Session(), context_secret_vault(ctx)
    )
    _echo_to_stderr(storage_notice(stored))
    _echo_to_stderr(f"Signed in. Continuing `{ctx.command_path}`.")
    return credential.access_token


def _proxy_accepts(base_url: str, api_key: str) -> bool:
    try:
        response: Final = requests.get(
            f"{base_url.rstrip('/')}/key/info", headers={"Authorization": f"Bearer {api_key}"}, timeout=10
        )
    except requests.RequestException:
        return False
    return response.ok or response.status_code == _SIGNED_IN_LOGIN_WITHOUT_KEY_ROW


def _sign_in_lock() -> SignInLock:
    return FileLock(Path(get_cli_token_file_path()).with_name("sign-in.lock"))


def _try_to_lock(lock: SignInLock, timeout: float) -> Literal["held", "busy", "unlockable"]:
    try:
        lock.acquire(timeout=timeout)
    except Timeout:
        return "busy"
    except OSError:
        return "unlockable"
    return "held"


def _lock_sign_in(lock: SignInLock) -> Literal["held", "waited", "unlockable"]:
    first_try: Final = _try_to_lock(lock, timeout=0)
    if first_try != "busy":
        return first_try
    _echo_to_stderr(
        "Another lite command is waiting for a browser sign-in, so this one waits for it instead of opening a second one."
    )
    match _try_to_lock(lock, timeout=_WAIT_FOR_ANOTHER_SIGN_IN_SECONDS):
        case "held":
            return "waited"
        case "unlockable":
            return "unlockable"
        case "busy":
            raise click.ClickException(
                "Another lite command has been waiting for a browser sign-in for over "
                f"{_WAIT_FOR_ANOTHER_SIGN_IN_SECONDS // 60} minutes, so this one stopped waiting. "
                "Finish or stop that sign-in, then run this command again."
            )


@contextmanager
def _sign_in_lock_held(lock: SignInLock) -> Generator[bool]:
    outcome: Final = _lock_sign_in(lock)
    if outcome == "unlockable":
        yield False
        return
    try:
        yield outcome == "waited"
    finally:
        lock.release()


def _login_for_another_proxy(ctx: click.Context, base_url: str) -> str | None:
    login: Final = load_cli_token(vault=context_secret_vault(ctx))
    if login is None or not login.base_url or login.base_url == base_url.rstrip("/"):
        return None
    return login.base_url


def _rerun_after_sign_in(run: Callable[[click.Context], object], ctx: click.Context, context: _SignInContext) -> object:
    base_url: Final = context["base_url"]
    if not context.get("api_key_from_token_file", False):
        raise click.ClickException(
            f"The proxy at {base_url} refused `{ctx.command_path}` with HTTP 401 for the API key from --api-key "
            "or LITELLM_PROXY_API_KEY. Check that the key is valid and has the role this command needs, "
            "or unset both to sign in through your browser instead."
        )
    sign_in: Final = context.get("browser_sign_in") or PkceBrowserSignIn()
    unavailable: Final = sign_in.unavailable_because()
    if unavailable is not None:
        _echo_to_stderr(f"The proxy at {base_url} needs a sign-in.")
        raise click.ClickException(_sign_in_failure(base_url, unavailable))
    rejected_key: Final = context.get("api_key")
    if rejected_key is not None and _proxy_accepts(base_url, rejected_key):
        raise click.ClickException(
            f"The proxy at {base_url} accepts your stored login but refused `{ctx.command_path}` with HTTP 401, "
            "so it likely lacks the role this command needs. Set LITELLM_PROXY_API_KEY to a key that has it."
        )
    with _sign_in_lock_held(context.get("sign_in_lock", _sign_in_lock)()) as waited_for_another_sign_in:
        return _rerun_once_signed_in(run, ctx, base_url, sign_in, rejected_key, waited_for_another_sign_in)


def _rerun_once_signed_in(
    run: Callable[[click.Context], object],
    ctx: click.Context,
    base_url: str,
    sign_in: BrowserSignIn,
    rejected_key: str | None,
    waited_for_another_sign_in: bool,
) -> object:
    retried: Final = _attempt_with_newer_stored_key(run, ctx, base_url, rejected_key)
    if isinstance(retried, _Completed):
        return retried.value
    other_proxy: Final = _login_for_another_proxy(ctx, base_url)
    if other_proxy is not None:
        raise click.ClickException(
            f"The proxy at {base_url} needs a sign-in, but lite keeps one stored login and it is for {other_proxy}, "
            f"so lite did not replace it on its own. Run `lite --base-url {base_url} login --pkce` to switch proxies, "
            "or set LITELLM_PROXY_API_KEY."
        )
    if waited_for_another_sign_in:
        raise click.ClickException(
            "The browser sign-in another lite command started ended while this one waited, "
            f"and the proxy at {base_url} still refuses `{ctx.command_path}`, so lite did not open a second one. "
            "Run the command again to sign in."
        )
    match _attempt_with_key(run, ctx, _sign_in(ctx, base_url, sign_in)):
        case _Completed(value=value):
            return value
        case _Unauthorized():
            raise click.ClickException(
                f"The proxy at {base_url} refused `{ctx.command_path}` with HTTP 401 again after the browser sign-in, "
                "so the signed-in user likely lacks the role this command needs. "
                "Set LITELLM_PROXY_API_KEY to a key that has it."
            )


class SignInOnUnauthorizedCommand(click.Command):
    def invoke(self, ctx: click.Context) -> object:
        run: Final = super().invoke
        match _attempt(run, ctx):
            case _Completed(value=value):
                return value
            case _Unauthorized(error=error):
                context: Final = _sign_in_context(ctx)
                if context is None:
                    raise error
                return _rerun_after_sign_in(run, ctx, context)


class SignInOnUnauthorizedGroup(click.Group):
    command_class = SignInOnUnauthorizedCommand
