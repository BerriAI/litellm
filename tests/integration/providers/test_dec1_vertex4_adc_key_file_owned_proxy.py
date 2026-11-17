"""Vertex credentials from `GOOGLE_APPLICATION_CREDENTIALS` and from `os.environ/` JSON text, chat and messages.

When a Vertex deployment names no `vertex_credentials` (and `VERTEXAI_CREDENTIALS` is unset), Python's
`VertexBase.load_auth` (`litellm/llms/vertex_ai/vertex_llm_base.py:196-235`) calls `google.auth.default()`
with the `cloud-platform` scope. The first Application Default Credentials source is the key file named by
`GOOGLE_APPLICATION_CREDENTIALS`, and the project it reports (the file's `project_id`) is used when no
`vertex_project` is set (`:220-225`). The token cache key for this case is the key file's path plus its
inode, mtime and size (`_credential_cache_identity`, `:110-117`), so rotating that file mints with the new key.

This module boots its own proxy whose environment holds `GOOGLE_APPLICATION_CREDENTIALS` (a generated
service account whose `token_uri` is a loopback peer), an empty `CLOUDSDK_CONFIG`, a `VERTEX_LOCATION`
(without `VERTEXAI_LOCATION`) and a variable holding service account JSON text for an `os.environ/` reference.
Every deployment is registered per test with `api_base` on its own peer, so nothing reaches a Google host.

The same inputs are exercised on `/v1/messages` (Vertex Claude), which #51294 ports to Rust:
`credential-gaps.md` rows 7 (`GOOGLE_APPLICATION_CREDENTIALS`) and 2 (inline JSON) say Rust keeps parity,
and the project table says Rust reads the ADC key file's `project_id`. One divergence between Python routes
is pinned on purpose: Gemini chat reads only `VERTEXAI_LOCATION` (`litellm/main.py:3925-3930`) while the
messages route also reads `VERTEX_LOCATION` (`vertex_llm_base.py:1406-1424`).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest
from integration._support.client import Gateway, gateway_from_environment
from integration._support.process import graceful_stop_seconds, owned_proxy
from integration._support.wire import Reply
from integration.providers.dec1_vertex4_support import (
    ADC_PROJECT,
    CLEARED_GOOGLE_ENVIRONMENT,
    REF_PROJECT,
    VERTEX_LOCATION_ENV,
    bearer,
    jwt_header,
    owned_config,
    vertex_echo,
)
from integration.providers.dec1_vertex_support import (
    CLAUDE,
    JWT_BEARER,
    PROJECT,
    SCOPE,
    Peer,
    claude_path,
    form_of,
    jwt_claims,
    marker,
    models_path,
    path_of,
    peer,
    service_account,
    service_account_text,
    vertex,
)

pytestmark: Final = [pytest.mark.timeout(2 * graceful_stop_seconds() + 240)]

_KEY_REF: Final = "os.environ/DEC1_VERTEX4_KEY_JSON"


def _no_model_host(_request: object) -> Reply:
    return Reply(status=404, body=b'{"error": "token peer only"}')


@pytest.fixture(scope="module")
def adc_peer() -> Iterator[Peer]:
    """The token endpoint of the service account `GOOGLE_APPLICATION_CREDENTIALS` names."""
    with peer(_no_model_host) as owned:
        yield owned


@pytest.fixture(scope="module")
def ref_peer() -> Iterator[Peer]:
    """The token endpoint of the service account held as JSON text in `DEC1_VERTEX4_KEY_JSON`."""
    with peer(_no_model_host) as owned:
        yield owned


@pytest.fixture(scope="module")
def adc_key_file(tmp_path_factory: pytest.TempPathFactory, adc_peer: Peer) -> Path:
    key: Final = tmp_path_factory.mktemp("dec1-vertex4-adc-key") / "adc-key.json"
    key.write_text(json.dumps(service_account(adc_peer.url, project=ADC_PROJECT)))
    return key


@pytest.fixture(scope="module")
def adc_proxy(tmp_path_factory: pytest.TempPathFactory, adc_key_file: Path, ref_peer: Peer) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("dec1-vertex4-adc")
    gcloud: Final = directory / "empty-gcloud-config"
    gcloud.mkdir()
    overrides: Final = {
        "GOOGLE_APPLICATION_CREDENTIALS": str(adc_key_file),
        "CLOUDSDK_CONFIG": str(gcloud),
        "VERTEX_LOCATION": VERTEX_LOCATION_ENV,
        "DEC1_VERTEX4_KEY_JSON": service_account_text(ref_peer.url, project=REF_PROJECT),
        "GOOGLE_AUTH_TRUST_BOUNDARY_ENABLED": "false",
        "NO_GCE_CHECK": "True",
    }
    removed: Final = tuple(name for name in CLEARED_GOOGLE_ENVIRONMENT if name not in overrides)
    with (
        gateway_from_environment() as gateway,
        owned_proxy(
            gateway, directory, overrides, config=owned_config(directory), remove_environment=removed, workers=1
        ) as proxy,
    ):
        yield proxy


def _messages() -> dict[str, object]:
    return {"max_tokens": 32, "messages": [{"role": "user", "content": f"adc {marker()}"}]}


# ---------------------------------------------------------------- Gemini chat


def test_gemini_chat_without_credentials_mints_with_the_application_credentials_key(
    adc_proxy: Gateway, adc_peer: Peer
) -> None:
    """A Gemini deployment with no `vertex_credentials` gets its token from the key file
    `GOOGLE_APPLICATION_CREDENTIALS` names: the key's token endpoint receives a JWT bearer grant, and the
    `generateContent` call carries a token that endpoint issued (`load_auth`, `vertex_llm_base.py:221`)."""
    with vertex(adc_proxy, vertex_echo, vertex_credentials=None) as rig:
        assert rig.chat().status_code == 200
        assert len(adc_peer.token_calls()) >= 1
        assert bearer(rig.peer.one_call()) in adc_peer.plan.issued


def test_the_application_credentials_grant_is_a_jwt_signed_by_the_key_file(adc_proxy: Gateway, adc_peer: Peer) -> None:
    """The grant posted to the key file's `token_uri` uses `grant_type=urn:ietf:params:oauth:grant-type:
    jwt-bearer`, and the assertion names the key file's `client_email` as issuer,
    `https://oauth2.googleapis.com/token` as audience and the `cloud-platform` scope google-auth was asked for
    (`vertex_llm_base.py:222`). google-auth signs the canonical Google token endpoint as the audience whatever
    the key's `token_uri` (google/oauth2/service_account.py, google-auth 2.52 in uv.lock)."""
    with vertex(adc_proxy, vertex_echo, vertex_credentials=None) as rig:
        assert rig.chat().status_code == 200
    grant: Final = form_of(adc_peer.token_calls()[0])
    assert grant["grant_type"] == JWT_BEARER
    claims: Final = jwt_claims(grant["assertion"])
    assert claims["iss"] == f"dec1-vertex@{ADC_PROJECT}.iam.gserviceaccount.com"
    assert claims["aud"] == "https://oauth2.googleapis.com/token"
    assert claims["scope"] == SCOPE


def test_gemini_chat_takes_the_project_from_the_application_credentials_key(adc_proxy: Gateway) -> None:
    """With neither `vertex_project` nor `VERTEXAI_PROJECT`, the project in the path is the key file's
    `project_id` that `google.auth.default()` reports (`vertex_llm_base.py:224-225`)."""
    with vertex(adc_proxy, vertex_echo, vertex_credentials=None, vertex_project=None) as rig:
        assert rig.chat().status_code == 200
        assert path_of(rig.peer.one_call()).startswith(f"/v1/projects/{ADC_PROJECT}/locations/")


def test_gemini_chat_vertex_project_wins_over_the_application_credentials_project(adc_proxy: Gateway) -> None:
    """A deployment `vertex_project` is used in the path even though the ADC key file names another
    project; the key only supplies the token (`load_auth` keeps a non-None `project_id`)."""
    with vertex(adc_proxy, vertex_echo, vertex_credentials=None) as rig:
        assert rig.chat().status_code == 200
        assert path_of(rig.peer.one_call()) == models_path(project=PROJECT) + ":generateContent"


def test_gemini_chat_ignores_the_vertex_location_variable(adc_proxy: Gateway) -> None:
    """Gemini chat reads only `VERTEXAI_LOCATION` (`litellm/main.py:3925-3930`), so with only
    `VERTEX_LOCATION=asia-northeast1` set and no `vertex_location` the call goes to the default
    `us-central1` (`get_vertex_region`, `vertex_llm_base.py:194`). The messages route reads the variable
    (see below); a Rust port that unifies the lookup changes this path and must do so on purpose."""
    with vertex(adc_proxy, vertex_echo, vertex_credentials=None, vertex_location=None) as rig:
        assert rig.chat().status_code == 200
        assert path_of(rig.peer.one_call()) == models_path(location="us-central1") + ":generateContent"


def test_one_application_credentials_token_serves_several_deployments(adc_proxy: Gateway, adc_peer: Peer) -> None:
    """Two Gemini deployments without credentials and with the same `vertex_project` share one cache entry
    (key file identity plus project, `get_access_token`, `vertex_llm_base.py:1064-1065`): three calls across
    both mint at most one token after the first, and every call carries an issued token."""
    with (
        vertex(adc_proxy, vertex_echo, vertex_credentials=None) as first,
        vertex(adc_proxy, vertex_echo, vertex_credentials=None) as second,
    ):
        assert first.chat().status_code == 200
        minted: Final = len(adc_peer.token_calls())
        assert second.chat().status_code == 200
        assert first.chat().status_code == 200
        assert len(adc_peer.token_calls()) == minted
        tokens: Final = {bearer(call) for call in (*first.peer.model_calls(), *second.peer.model_calls())}
        assert tokens <= set(adc_peer.plan.issued)


def test_a_deployment_with_its_own_credentials_never_uses_the_application_credentials(
    adc_proxy: Gateway, adc_peer: Peer
) -> None:
    """A deployment that names `vertex_credentials` mints at its own token endpoint; the ADC key file's
    endpoint gets no new grant and the call does not carry an ADC token (`load_auth` only falls back to
    `google.auth.default()` when credentials are None)."""
    before: Final = len(adc_peer.token_calls())
    with vertex(adc_proxy, vertex_echo) as rig:
        assert rig.chat().status_code == 200
        assert len(adc_peer.token_calls()) == before
        assert bearer(rig.peer.one_call()) in rig.peer.plan.issued


# ---------------------------------------------------------------- os.environ/ holding JSON text


def test_an_os_environ_reference_holding_json_text_is_parsed_as_the_key(adc_proxy: Gateway, ref_peer: Peer) -> None:
    """`vertex_credentials: os.environ/DEC1_VERTEX4_KEY_JSON` whose variable holds service account JSON text
    (not a path) is resolved at registration and loaded as inline JSON (`credential-gaps.md` row 2): the
    token comes from that key's endpoint and the path uses its `project_id` when `vertex_project` is unset."""
    with vertex(adc_proxy, vertex_echo, vertex_credentials=_KEY_REF, vertex_project=None) as rig:
        assert rig.chat().status_code == 200
        call: Final = rig.peer.one_call()
        assert bearer(call) in ref_peer.plan.issued
        assert path_of(call) == models_path(project=REF_PROJECT) + ":generateContent"


def test_messages_with_an_os_environ_json_reference_mints_from_that_key(adc_proxy: Gateway, ref_peer: Peer) -> None:
    """The same `os.environ/` JSON reference on a Vertex Claude deployment authenticates `/v1/messages`:
    the rawPredict call carries a token from the referenced key's endpoint and names its `project_id`."""
    with vertex(adc_proxy, vertex_echo, backend=CLAUDE, vertex_credentials=_KEY_REF, vertex_project=None) as rig:
        assert rig.post("/v1/messages", _messages()).status_code == 200
        call: Final = rig.peer.one_call()
        assert bearer(call) in ref_peer.plan.issued
        assert path_of(call) == claude_path(project=REF_PROJECT) + ":rawPredict"


# ---------------------------------------------------------------- /v1/messages (Vertex Claude)


def test_messages_without_credentials_mints_with_the_application_credentials_key(
    adc_proxy: Gateway, adc_peer: Peer
) -> None:
    """`/v1/messages` to a Vertex Claude deployment with no credentials authenticates with a token from the
    `GOOGLE_APPLICATION_CREDENTIALS` key (`validate_anthropic_messages_environment` passes credentials None
    to `get_access_token`, `anthropic/messages/transformation.py:117-123`). `credential-gaps.md` row 7."""
    with vertex(adc_proxy, vertex_echo, backend=CLAUDE, vertex_credentials=None) as rig:
        assert rig.post("/v1/messages", _messages()).status_code == 200
        assert bearer(rig.peer.one_call()) in adc_peer.plan.issued


def test_messages_takes_the_project_from_the_application_credentials_key(adc_proxy: Gateway) -> None:
    """With no `vertex_project`, the rawPredict path names the ADC key file's `project_id`
    (`_finalize_vertex_messages_environment` passes the resolved `project_id`, `transformation.py:183-191`).
    The Rust port reads the same file (`credential-gaps.md`, project table, ADC key file row)."""
    with vertex(adc_proxy, vertex_echo, backend=CLAUDE, vertex_credentials=None, vertex_project=None) as rig:
        assert rig.post("/v1/messages", _messages()).status_code == 200
        assert path_of(rig.peer.one_call()).startswith(f"/v1/projects/{ADC_PROJECT}/locations/")


def test_messages_reads_the_vertex_location_variable(adc_proxy: Gateway) -> None:
    """On `/v1/messages` a deployment without `vertex_location` uses `VERTEX_LOCATION=asia-northeast1`
    (`safe_get_vertex_ai_location`, `vertex_llm_base.py:1419-1424`), so the call goes to
    `.../locations/asia-northeast1/publishers/anthropic/models/<model>:rawPredict`. UNVERIFIED for the Rust
    port: `credential-gaps.md` does not list `VERTEX_LOCATION` among the names Rust reads."""
    with vertex(adc_proxy, vertex_echo, backend=CLAUDE, vertex_credentials=None, vertex_location=None) as rig:
        assert rig.post("/v1/messages", _messages()).status_code == 200
        assert path_of(rig.peer.one_call()) == claude_path(location=VERTEX_LOCATION_ENV) + ":rawPredict"


def test_streamed_messages_carry_the_application_credentials_token(adc_proxy: Gateway, adc_peer: Peer) -> None:
    """A streamed `/v1/messages` call with ADC credentials posts to `:streamRawPredict` with a bearer issued
    by the ADC key's endpoint, the same token source as the unary call."""
    with vertex(adc_proxy, vertex_echo, backend=CLAUDE, vertex_credentials=None) as rig:
        response, _text = rig.stream("/v1/messages", _messages())
        assert response.status_code == 200
        call: Final = rig.peer.one_call()
        assert path_of(call) == claude_path() + ":streamRawPredict"
        assert bearer(call) in adc_peer.plan.issued


# ---------------------------------------------------------------- key file rotation (runs last: it rewrites the file)


def test_rotating_the_application_credentials_file_mints_with_the_new_key(
    adc_proxy: Gateway, adc_peer: Peer, adc_key_file: Path
) -> None:
    """The ADC cache key includes the key file's inode, mtime and size (`_credential_cache_identity`,
    `vertex_llm_base.py:113-117`), so after `GOOGLE_APPLICATION_CREDENTIALS`' file is replaced by a new
    key the next call mints again and the new grant is signed with the new `private_key_id` as JWT `kid`."""
    with vertex(adc_proxy, vertex_echo, vertex_credentials=None) as rig:
        assert rig.chat().status_code == 200
        before: Final = len(adc_peer.token_calls())
        rotated: Final = service_account(adc_peer.url, project=ADC_PROJECT, private_key_id="dec1-rotated-adc-key")
        adc_key_file.write_text(json.dumps(rotated, indent=2))
        assert rig.chat().status_code == 200
        grants: Final = adc_peer.token_calls()[before:]
        assert len(grants) >= 1
        assert jwt_header(form_of(grants[-1])["assertion"])["kid"] == "dec1-rotated-adc-key"
        assert bearer(rig.peer.model_calls()[-1]) in adc_peer.plan.issued[before:]
