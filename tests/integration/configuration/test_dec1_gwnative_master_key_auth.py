"""Master-key-only authentication on the native Rust gateway preview.

The contract: every mounted inference route (`/v1/messages`, `/chat/completions`, `/v1/responses`,
`/v1/embeddings`, ...) requires exactly one `Authorization: Bearer <master key>` header
(`gateway-auth/src/http.rs`). Missing, duplicate, empty, malformed, wrong and virtual-key tokens all
get the same 401 with the plain-text body `missing or invalid bearer token`, before the body is read
and before any provider is called. The comparison is on SHA-256 digests in constant time. The preview
has no key store: a virtual key is just a wrong token. The master key comes only from the config's
`general_settings.master_key` (itself possibly an `os.environ/` reference); `LITELLM_MASTER_KEY` alone
is not read, unlike the Python proxy.

One gateway serves the module; the boot-specific cases boot their own.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Final

import httpx
import pytest

from tests.integration.configuration.dec1_gwnative_support import (
    GATEWAY_KEY,
    UNAUTHENTICATED,
    UNCONFIGURED,
    NativeGateway,
    ScriptedProvider,
    anthropic_deployment,
    chat_body,
    marker,
    master_key_config,
    messages_body,
    native_gateway,
    openai_deployment,
    openai_like_deployment,
    responses_body,
    scripted_provider,
)

_ROUTES: Final = (
    "/v1/messages",
    "/chat/completions",
    "/v1/chat/completions",
    "/responses",
    "/v1/responses",
    "/embeddings",
    "/v1/embeddings",
    "/completions",
    "/v1/completions",
    "/ocr",
    "/v1/ocr",
    "/audio/transcriptions",
    "/v1/audio/transcriptions",
    "/openai/deployments/chat/chat/completions",
    "/engines/chat/chat/completions",
)


@pytest.fixture(scope="module")
def provider() -> Iterator[ScriptedProvider]:
    with scripted_provider() as scripted:
        yield scripted


@pytest.fixture(scope="module")
def gateway(provider: ScriptedProvider, tmp_path_factory: pytest.TempPathFactory) -> Iterator[NativeGateway]:
    config: Final = master_key_config(
        [
            anthropic_deployment("claude", provider.url),
            openai_like_deployment("chat", provider.url),
            openai_deployment("responses", provider.url),
        ]
    )
    with native_gateway(config, tmp_path_factory.mktemp("gwnative-auth")) as booted:
        yield booted


def _post(gateway: NativeGateway, path: str, headers: list[tuple[str, str]], body: dict[str, object]) -> httpx.Response:
    with gateway.client() as client:
        return client.post(path, json=body, headers=httpx.Headers(headers))


def test_the_master_key_as_bearer_is_served(gateway: NativeGateway, provider: ScriptedProvider) -> None:
    """`Authorization: Bearer <master key>` on `POST /v1/messages` for a configured model. The answer must be
    200 and the provider must have received exactly one request for this prompt."""
    case: Final = marker()
    response: Final = gateway.post("/v1/messages", messages_body(case))
    assert response.status_code == 200, response.text
    assert len(provider.requests_for(case)) == 1


@pytest.mark.parametrize("path", _ROUTES)
def test_every_inference_route_answers_401_without_credentials(gateway: NativeGateway, path: str) -> None:
    """`POST <path>` with no Authorization header. Every mounted inference route, including those the
    preview only answers 501 for, must answer 401 with the plain-text body `missing or invalid bearer
    token` (`Error::InvalidToken`), so an anonymous caller never learns which routes are implemented."""
    response: Final = gateway.post(path, {"model": "chat"}, key=None)
    assert response.status_code == 401, response.text
    assert response.text == UNAUTHENTICATED


def test_the_401_body_is_plain_text_not_an_openai_envelope(gateway: NativeGateway) -> None:
    """A missing token on `POST /v1/chat/completions`. The rejection is rendered by
    `gateway_auth::Error::into_response` as `(status, String)`, so the content type must start with
    `text/plain` and the body must be the bare message, not JSON. The Python proxy answers JSON here;
    clients that parse `error.message` will see a parse failure on the preview."""
    response: Final = gateway.post("/v1/chat/completions", chat_body("dec1"), key=None)
    assert response.status_code == 401
    assert response.headers["content-type"].startswith("text/plain"), response.headers
    assert response.text == UNAUTHENTICATED


@pytest.mark.parametrize(
    "authorization",
    [
        "Bearer sk-dec1-wrong-key",
        "bearer {key}",
        "BEARER {key}",
        "Basic {key}",
        "{key}",
        "Bearer",
        "Bearer {key} extra",
        "Bearer {key}\tx",
    ],
    ids=[
        "wrong_key",
        "lowercase_scheme",
        "uppercase_scheme",
        "basic_scheme",
        "bare_key",
        "scheme_only",
        "two_words",
        "inner_tab",
    ],
)
def test_malformed_or_wrong_authorization_is_401(
    gateway: NativeGateway, provider: ScriptedProvider, authorization: str
) -> None:
    """`Authorization: <variant>` on `POST /v1/messages`. The extractor only accepts the exact prefix
    `Bearer ` followed by one token with no inner whitespace (`bearer` in `gateway-auth/src/http.rs`), and
    the token must equal the master key. Each variant must get 401 with `missing or invalid bearer token`
    and the provider must see nothing. The Python proxy accepts a lowercase `bearer` scheme; that is a
    parity gap the preview has on purpose ("strict" extractor)."""
    case: Final = marker()
    response: Final = _post(
        gateway, "/v1/messages", [("Authorization", authorization.format(key=GATEWAY_KEY))], messages_body(case)
    )
    assert response.status_code == 401, response.text
    assert response.text == UNAUTHENTICATED
    assert provider.requests_for(case) == ()


def test_extra_spaces_between_scheme_and_key_are_trimmed(gateway: NativeGateway) -> None:
    """`Authorization: Bearer    <master key>` (several spaces after the scheme). The token is trimmed
    after the `Bearer ` prefix is removed, so the call must be served with 200."""
    response: Final = _post(
        gateway, "/v1/messages", [("Authorization", f"Bearer    {GATEWAY_KEY}")], messages_body(marker())
    )
    assert response.status_code == 200, response.text


def test_two_authorization_headers_are_refused_even_if_one_is_right(gateway: NativeGateway) -> None:
    """Two `Authorization` headers, the first with the master key and the second wrong. A duplicate header
    is always a 401 (`values.next().is_some()` check), so neither order may be served: an intermediary that
    appends a header cannot smuggle a second identity."""
    first_right: Final = _post(
        gateway,
        "/v1/messages",
        [("Authorization", f"Bearer {GATEWAY_KEY}"), ("Authorization", "Bearer sk-dec1-other")],
        messages_body(marker()),
    )
    both_right: Final = _post(
        gateway,
        "/v1/messages",
        [("Authorization", f"Bearer {GATEWAY_KEY}"), ("Authorization", f"Bearer {GATEWAY_KEY}")],
        messages_body(marker()),
    )
    assert (first_right.status_code, both_right.status_code) == (401, 401)


@pytest.mark.parametrize("header", ["x-api-key", "x-litellm-api-key", "api-key", "Ocp-Apim-Subscription-Key"])
def test_the_master_key_in_another_header_is_refused_today(gateway: NativeGateway, header: str) -> None:
    """Pins the preview: the master key sent as `<header>` with no Authorization header on
    `POST /v1/messages`. Only `Authorization: Bearer` is read (the One Pager lists `x-api-key` and
    `x-litellm-api-key` as not started), so the answer must be 401 `missing or invalid bearer token`.
    The Python proxy accepts these headers; see the parity file for the intended behaviour."""
    response: Final = gateway.post("/v1/messages", messages_body(marker()), key=None, headers={header: GATEWAY_KEY})
    assert response.status_code == 401, response.text
    assert response.text == UNAUTHENTICATED


def test_a_virtual_key_shaped_token_is_refused_like_any_wrong_token(
    gateway: NativeGateway, provider: ScriptedProvider
) -> None:
    """A token that looks like a generated virtual key (`sk-` plus 22 characters) on `POST /v1/messages`.
    The preview has no key store (`LocalAdministrator` only admits the master-key principal), so it must
    get the same 401 `missing or invalid bearer token` as any wrong token, with no provider call. It must not
    be a 500 or a "database not connected" error, so clients cannot tell a valid stored key from garbage."""
    case: Final = marker()
    response: Final = gateway.post("/v1/messages", messages_body(case), key="sk-dec1AbCdEfGhIjKlMnOpQrStU")
    assert response.status_code == 401, response.text
    assert response.text == UNAUTHENTICATED
    assert provider.requests_for(case) == ()


def test_a_rejection_never_echoes_the_presented_token(gateway: NativeGateway) -> None:
    """A wrong token `sk-dec1-echo-check-token` on `POST /v1/chat/completions`. The 401 body must not contain
    the token, and must not name the requested model either (the body is never read before auth)."""
    response: Final = gateway.post(
        "/v1/chat/completions", chat_body("dec1", "secret-model-name"), key="sk-dec1-echo-check-token"
    )
    assert response.status_code == 401
    assert "sk-dec1-echo-check-token" not in response.text
    assert "secret-model-name" not in response.text


def test_an_unauthenticated_oversized_body_is_refused_with_401_not_413(gateway: NativeGateway) -> None:
    """No credentials and a 52 MiB body on `POST /v1/chat/completions`. Authentication runs as a route layer
    before the handler extracts the body, so the answer must be 401, not 413: an anonymous caller cannot
    make the gateway buffer a large body."""
    content: Final = b'{"model": "chat", "pad": "' + b"a" * (52 * 1024 * 1024) + b'"}'
    response: Final = gateway.post("/v1/chat/completions", key=None, content=content)
    assert response.status_code == 401, response.text


def test_unknown_paths_are_404_whatever_the_credentials(gateway: NativeGateway) -> None:
    """`POST /dec1/not-a-route` with no header and with the master key. Unknown paths are not matched, so the
    route layer does not run: both must answer 404 (`unknown_path` cases in `gateway/tests/server.rs`)."""
    anonymous: Final = gateway.post("/dec1/not-a-route", {}, key=None)
    admin: Final = gateway.post("/dec1/not-a-route", {})
    assert (anonymous.status_code, admin.status_code) == (404, 404)


@pytest.mark.parametrize("path", ["/key/generate", "/team/new", "/user/new", "/model/new", "/config/update"])
def test_management_writes_are_404_for_the_master_key(gateway: NativeGateway, path: str) -> None:
    """`POST <path>` with the master key. The preview has no management API and no database, so these must
    answer 404. A preview user cannot create virtual keys, teams or users; there is no partial write."""
    response: Final = gateway.post(path, {"models": ["chat"]})
    assert response.status_code == 404, response.text


def test_master_key_from_an_os_environ_reference_in_the_file(tmp_path: Path) -> None:
    """`general_settings.master_key: os.environ/DEC1_GW_MASTER` and the process has that variable. A call with
    that value must be served (200) and a call with the value of `LITELLM_MASTER_KEY` refused (401): the
    reference is resolved by `load_config` and is the only key."""
    with scripted_provider() as provider:
        config: Final = {
            "model_list": [anthropic_deployment("claude", provider.url)],
            "general_settings": {"master_key": "os.environ/DEC1_GW_MASTER"},
        }
        with native_gateway(config, tmp_path, {"DEC1_GW_MASTER": "sk-dec1-gw-referenced-master"}) as gateway:
            referenced: Final = gateway.post(
                "/v1/messages", messages_body(marker()), key="sk-dec1-gw-referenced-master"
            )
            standard: Final = gateway.post("/v1/messages", messages_body(marker()), key=GATEWAY_KEY)
    assert (referenced.status_code, standard.status_code) == (200, 401), (referenced.text, standard.text)


def test_a_literal_master_key_in_the_file_is_accepted(tmp_path: Path) -> None:
    """`general_settings.master_key: sk-dec1-gw-literal` written in the file. That literal must be served with
    200 on `POST /v1/messages`."""
    with scripted_provider() as provider:
        config: Final = {
            "model_list": [anthropic_deployment("claude", provider.url)],
            "general_settings": {"master_key": "sk-dec1-gw-literal"},
        }
        with native_gateway(config, tmp_path) as gateway:
            response: Final = gateway.post("/v1/messages", messages_body(marker()), key="sk-dec1-gw-literal")
    assert response.status_code == 200, response.text


def test_litellm_master_key_alone_is_not_read_by_the_preview(tmp_path: Path) -> None:
    """Pins the preview: the config has no `master_key` while `LITELLM_MASTER_KEY` is set in the process.
    `Auth::from_config` reads only `general_settings.master_key`, so a call with the environment key must
    answer 500 `gateway auth not configured`. The Python proxy falls back to `LITELLM_MASTER_KEY` and
    would answer 200; a preview config must say `master_key: os.environ/LITELLM_MASTER_KEY`."""
    with scripted_provider() as provider:
        config: Final = {"model_list": [anthropic_deployment("claude", provider.url)]}
        with native_gateway(config, tmp_path) as gateway:
            response: Final = gateway.post("/v1/messages", messages_body(marker()))
    assert response.status_code == 500, response.text
    assert response.text == UNCONFIGURED


def test_a_blank_master_key_is_treated_as_unconfigured(tmp_path: Path) -> None:
    """`general_settings.master_key: "   "` (only spaces). `MasterKeyAuthenticator::key` filters out a key
    that trims to empty, so `POST /v1/messages` with any bearer token must answer 500
    `gateway auth not configured`, never accept an empty credential."""
    with scripted_provider() as provider:
        config: Final = {
            "model_list": [anthropic_deployment("claude", provider.url)],
            "general_settings": {"master_key": "   "},
        }
        with native_gateway(config, tmp_path) as gateway:
            response: Final = gateway.post("/v1/messages", messages_body(marker()), key="anything")
    assert response.status_code == 500, response.text
    assert response.text == UNCONFIGURED


def test_an_unresolvable_master_key_reference_is_unconfigured(tmp_path: Path) -> None:
    """`master_key: os.environ/DEC1_GW_MISSING_MASTER` and the variable is not set. A missing `os.environ/`
    reference resolves to null, so the gateway boots and every inference call answers 500
    `gateway auth not configured`. UNVERIFIED: whether `litellm_config_load` turns a missing reference into
    null or refuses the boot; check `config-load/src` if this fails with a boot refusal."""
    config: Final = {"model_list": [], "general_settings": {"master_key": "os.environ/DEC1_GW_MISSING_MASTER"}}
    with native_gateway(config, tmp_path) as gateway:
        response: Final = gateway.post("/v1/messages", messages_body(marker()), key="anything")
    assert response.status_code == 500, response.text
    assert response.text == UNCONFIGURED


def test_the_responses_route_is_authenticated_like_messages(gateway: NativeGateway, provider: ScriptedProvider) -> None:
    """`POST /v1/responses` with a wrong key and then with the master key. The wrong key must get 401 with
    no provider call; the master key must reach the provider and get 200."""
    case: Final = marker()
    refused: Final = gateway.post("/v1/responses", responses_body(case), key="sk-dec1-wrong")
    assert refused.status_code == 401
    assert provider.requests_for(case) == ()
    served: Final = gateway.post("/v1/responses", responses_body(case))
    assert served.status_code == 200, served.text
    assert len(provider.requests_for(case)) == 1
