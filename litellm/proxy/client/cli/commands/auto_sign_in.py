import webbrowser
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Literal, Protocol

import click
import requests
from typing_extensions import NotRequired, ReadOnly, TypedDict

from litellm.litellm_core_utils.cli_token_utils import load_cli_token

from ...exceptions import UnauthorizedError
from .auth import context_secret_vault, get_stored_api_key, replace_stored_token, storage_notice
from .pkce_login import PkceCredential, PkceFailure, pkce_token_record, run_pkce_login


class BrowserSignIn(Protocol):
    def __call__(self, base_url: str, echo: Callable[[str], None]) -> PkceCredential | PkceFailure: ...


def pkce_browser_sign_in(
    base_url: str, echo: Callable[[str], None], find_browser: Callable[[], object] = webbrowser.get
) -> PkceCredential | PkceFailure:
    try:
        _ = find_browser()
    except webbrowser.Error:
        return PkceFailure("no web browser is available on this machine")
    return run_pkce_login(base_url, requests.Session(), echo=echo)


class _SignInContext(TypedDict):
    base_url: ReadOnly[str]
    api_key: ReadOnly[NotRequired[str | None]]
    api_key_from_token_file: ReadOnly[NotRequired[bool]]
    browser_sign_in: ReadOnly[NotRequired[BrowserSignIn]]


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
    _echo_to_stderr(f"The proxy at {base_url} needs a sign-in, so lite is opening your browser to sign in.")
    sign_in: Final = context.get("browser_sign_in") or pkce_browser_sign_in
    credential: Final = sign_in(base_url, _echo_to_stderr)
    if isinstance(credential, PkceFailure):
        raise click.ClickException(
            f"Could not sign in to {base_url} through the browser: {credential.reason}. "
            "Run `lite login` from a terminal, or set LITELLM_PROXY_API_KEY."
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
            f"{base_url.rstrip('/')}/v1/models", headers={"Authorization": f"Bearer {api_key}"}, timeout=10
        )
    except requests.RequestException:
        return False
    return response.ok


def _login_for_another_proxy(ctx: click.Context, base_url: str) -> str | None:
    login: Final = load_cli_token(vault=context_secret_vault(ctx))
    if login is None or not login.base_url or login.base_url == base_url.rstrip("/"):
        return None
    return login.base_url


def _rerun_after_sign_in(run: Callable[[click.Context], object], ctx: click.Context, context: _SignInContext) -> object:
    base_url: Final = context["base_url"]
    rejected_key: Final = context.get("api_key")
    if rejected_key is not None and _proxy_accepts(base_url, rejected_key):
        raise click.ClickException(
            f"The proxy at {base_url} accepts this API key but refused `{ctx.command_path}` with HTTP 401, "
            "so the key likely lacks the role this command needs."
        )
    if not context.get("api_key_from_token_file", False):
        raise click.ClickException(
            f"The proxy at {base_url} rejected the API key from --api-key or LITELLM_PROXY_API_KEY. "
            "Check that key, or unset both to sign in through your browser instead."
        )
    retried: Final = _attempt_with_newer_stored_key(run, ctx, base_url, rejected_key)
    if isinstance(retried, _Completed):
        return retried.value
    other_proxy: Final = _login_for_another_proxy(ctx, base_url)
    if other_proxy is not None:
        raise click.ClickException(
            f"The proxy at {base_url} needs a sign-in, but lite keeps one stored login and it is for {other_proxy}, "
            f"so lite did not replace it on its own. Run `lite --base-url {base_url} login` to switch proxies, "
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
