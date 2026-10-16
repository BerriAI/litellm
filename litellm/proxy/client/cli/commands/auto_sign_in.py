import os
import subprocess
import sys
import webbrowser
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Protocol

import click
import requests
from filelock import FileLock
from typing_extensions import NotRequired, ReadOnly, TypedDict

from litellm.litellm_core_utils.cli_token_utils import get_cli_token_file_path, load_cli_token

from ...exceptions import UnauthorizedError
from .auth import context_secret_vault, get_stored_api_key, replace_stored_token, storage_notice
from .pkce_login import LOGIN_TIMEOUT_SECONDS, PkceCredential, PkceFailure, pkce_token_record, run_pkce_login


class BrowserSignIn(Protocol):
    def __call__(self, base_url: str, echo: Callable[[str], None]) -> PkceCredential | PkceFailure: ...


_TEXT_MODE_BROWSERS: Final = frozenset({"www-browser", "links", "elinks", "lynx", "w3m"})
_SSH_SESSION_VARIABLES: Final = ("SSH_CONNECTION", "SSH_TTY")
_DISPLAY_VARIABLES: Final = ("DISPLAY", "WAYLAND_DISPLAY")


def _runs_in_ci(environ: Mapping[str, str]) -> bool:
    return environ.get("CI", "").strip().lower() not in {"", "0", "false"}


def _ssh_session_without_a_display(environ: Mapping[str, str]) -> bool:
    over_ssh: Final = any(environ.get(name) for name in _SSH_SESSION_VARIABLES)
    return over_ssh and not any(environ.get(name) for name in _DISPLAY_VARIABLES)


_OPEN_URL_IN_BROWSER: Final = "import sys, webbrowser; sys.exit(0 if webbrowser.open(sys.argv[1]) else 1)"


def open_browser_off_stdout(url: str) -> PkceFailure | None:
    launcher: Final = subprocess.run((sys.executable, "-I", "-c", _OPEN_URL_IN_BROWSER, url), stdout=2, check=False)
    return None if launcher.returncode == 0 else PkceFailure("the web browser did not open")


def pkce_browser_sign_in(
    base_url: str,
    echo: Callable[[str], None],
    find_browser: Callable[[], webbrowser.BaseBrowser] = webbrowser.get,
    environ: Mapping[str, str] = os.environ,
    open_browser: Callable[[str], object] = open_browser_off_stdout,
) -> PkceCredential | PkceFailure:
    if _runs_in_ci(environ):
        return PkceFailure("the CI environment variable is set, so lite does not open a browser")
    if "BROWSER" not in environ and _ssh_session_without_a_display(environ):
        return PkceFailure("this is an SSH session without a display, so a browser would open where you cannot see it")
    try:
        browser: Final = find_browser()
    except webbrowser.Error:
        return PkceFailure("no web browser is available on this machine")
    if browser.name in _TEXT_MODE_BROWSERS and "BROWSER" not in environ:
        return PkceFailure("only a text-mode web browser is available on this machine")
    echo(
        f"lite is opening your browser to sign in and waits up to {LOGIN_TIMEOUT_SECONDS // 60} minutes "
        "for you to approve it. A shell or agent that stops commands sooner should run this one with a longer timeout."
    )
    return run_pkce_login(base_url, requests.Session(), open_browser=open_browser, echo=echo)


class _SignInContext(TypedDict):
    base_url: ReadOnly[str]
    api_key: ReadOnly[NotRequired[str | None]]
    api_key_from_token_file: ReadOnly[NotRequired[bool]]
    browser_sign_in: ReadOnly[NotRequired[BrowserSignIn]]
    sign_in_lock: ReadOnly[NotRequired[Callable[[], AbstractContextManager[object]]]]


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


def _sign_in(ctx: click.Context, context: _SignInContext) -> str:
    base_url: Final = context["base_url"]
    _echo_to_stderr(f"The proxy at {base_url} needs a sign-in.")
    sign_in: Final = context.get("browser_sign_in") or pkce_browser_sign_in
    credential: Final = sign_in(base_url, _echo_to_stderr)
    if isinstance(credential, PkceFailure):
        raise click.ClickException(
            f"Could not sign in to {base_url} through the browser: {credential.reason}. "
            "Run `lite login --pkce` from a terminal on a machine with a browser, `lite login` on one without, "
            "or set LITELLM_PROXY_API_KEY."
        )
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
    return response.ok


def _sign_in_lock() -> FileLock:
    return FileLock(Path(get_cli_token_file_path()).with_name("sign-in.lock"))


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
    rejected_key: Final = context.get("api_key")
    if rejected_key is not None and _proxy_accepts(base_url, rejected_key):
        raise click.ClickException(
            f"The proxy at {base_url} accepts your stored login but refused `{ctx.command_path}` with HTTP 401, "
            "so it likely lacks the role this command needs. Set LITELLM_PROXY_API_KEY to a key that has it."
        )
    with context.get("sign_in_lock", _sign_in_lock)():
        return _rerun_once_signed_in(run, ctx, context, rejected_key)


def _rerun_once_signed_in(
    run: Callable[[click.Context], object], ctx: click.Context, context: _SignInContext, rejected_key: str | None
) -> object:
    base_url: Final = context["base_url"]
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
    match _attempt_with_key(run, ctx, _sign_in(ctx, context)):
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
