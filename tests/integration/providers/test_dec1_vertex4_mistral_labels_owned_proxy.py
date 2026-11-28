"""Billing labels for Mistral and Codestral on Vertex when `vertex_request_label_fields` is configured.

Vertex partner models that speak the OpenAI wire format take billing labels in the base64 JSON header
`X-Vertex-AI-Labels`, not in the body. In `VertexAIPartnerModels.completion`
(`litellm/llms/vertex_ai/vertex_ai_partner_models/main.py:178-197`) only the Mistral family (any model whose
name contains `mistral` or `codestral`) gets that header, and only when `litellm.vertex_request_label_fields`
is non-empty: the caller's `labels` (and `extra_body.labels`) are removed from the body and resolved by
`vertex_partner_labels_header` (`litellm/llms/vertex_ai/request_labels.py`): identity fields first (the
`user_api_key_` prefix is reserved and normalized), then the caller's string labels verbatim, then
`spend_logs_metadata` pairs normalized (lowercase, disallowed characters to `_`), each key keeping its first
value, at most 32 labels (`VERTEX_PARTNER_REQUEST_LABELS_MAX`). A decodable labels header the deployment
already carries stands in for the caller's body labels.

Round 3 listed this as not covered because the shared proxy does not set the option, so this module boots
its own proxy with `vertex_request_label_fields: [user_api_key_alias, spend_logs_metadata]`. Gemini and
Claude labels on such a proxy are already pinned in `test_vertex_request_label_identity_wire.py`.
"""

from __future__ import annotations

import base64
import json
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final

import pytest
from integration._support.client import Gateway, gateway_from_environment
from integration._support.process import graceful_stop_seconds, owned_proxy
from integration.providers.dec1_vertex3_support import LLAMA, MISTRAL, openai_echo
from integration.providers.dec1_vertex4_support import labels_header, owned_config
from integration.providers.dec1_vertex_support import Rig, body_of, marker, vertex
from pydantic import JsonValue

pytestmark: Final = [pytest.mark.timeout(2 * graceful_stop_seconds() + 180)]

_CODESTRAL_CHAT: Final = "codestral-2501@001"


@dataclass(frozen=True, slots=True)
class Labelled:
    proxy: Gateway
    key: str
    alias: str
    bare_key: str


@pytest.fixture(scope="module")
def labelled(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Labelled]:
    directory: Final = tmp_path_factory.mktemp("dec1-vertex4-labels")
    config: Final = owned_config(directory, vertex_request_label_fields=["user_api_key_alias", "spend_logs_metadata"])
    alias: Final = f"Billing Key {uuid.uuid4().hex[:8]}"
    with (
        gateway_from_environment() as gateway,
        owned_proxy(gateway, directory, {}, config=config, workers=1) as proxy,
        proxy.scenario() as scenario,
    ):
        yield Labelled(
            proxy=proxy,
            key=scenario.key(
                key_alias=alias, metadata={"spend_logs_metadata": {"Cost Center": "CC 12", "owner": "data"}}
            ),
            alias=alias,
            bare_key=scenario.key(),
        )


def _expected_alias(alias: str) -> str:
    """The alias as a label value: lowercased, spaces to `_` (`_normalized`, `request_labels.py`)."""
    return alias.lower().replace(" ", "_")


def _chat(labelled: Labelled, rig: Rig, key: str | None = None, **fields: JsonValue) -> None:
    body: Final[dict[str, JsonValue]] = {
        "model": rig.model,
        "messages": [{"role": "user", "content": f"labels {marker()}"}],
        **fields,
    }
    response: Final = labelled.proxy.request("POST", "/v1/chat/completions", body, key=key or labelled.key)
    assert response.status_code == 200, response.text


def test_mistral_sends_the_key_alias_as_a_labels_header(labelled: Labelled) -> None:
    """A Mistral call from a key aliased `Billing Key <id>` carries `X-Vertex-AI-Labels` decoding to an
    object whose first label is `user_api_key_alias: billing_key_<id>`, followed by the key's
    `spend_logs_metadata` pairs."""
    with vertex(labelled.proxy, openai_echo, backend=MISTRAL) as rig:
        _chat(labelled, rig)
        labels: Final = labels_header(rig.peer.one_call())
        assert labels is not None
        assert next(iter(labels.items())) == ("user_api_key_alias", _expected_alias(labelled.alias))


def test_mistral_spend_logs_metadata_becomes_normalized_labels(labelled: Labelled) -> None:
    """The key's `spend_logs_metadata` `{"Cost Center": "CC 12", "owner": "data"}` is sent as
    `cost_center: cc_12` and `owner: data`: keys and values are lowercased and the space becomes `_`
    (`_spend_logs_labels`, `request_labels.py`)."""
    with vertex(labelled.proxy, openai_echo, backend=MISTRAL) as rig:
        _chat(labelled, rig)
        labels: Final = labels_header(rig.peer.one_call())
        assert labels is not None
        assert labels.get("cost_center") == "cc_12"
        assert labels.get("owner") == "data"


def test_mistral_caller_labels_move_from_the_body_into_the_header(labelled: Labelled) -> None:
    """A caller's `labels: {"Team": "Search Team"}` is removed from the rawPredict body (`main.py:190-191`)
    and appears verbatim in the header after the identity label; caller labels are not normalized."""
    with vertex(labelled.proxy, openai_echo, backend=MISTRAL) as rig:
        _chat(labelled, rig, labels={"Team": "Search Team"})
        call: Final = rig.peer.one_call()
        assert "labels" not in body_of(call)
        labels: Final = labels_header(call)
        assert labels is not None
        assert labels.get("Team") == "Search Team"
        assert next(iter(labels)) == "user_api_key_alias"


def test_mistral_extra_body_labels_also_move_into_the_header(labelled: Labelled) -> None:
    """Labels sent as `extra_body: {"labels": {"project": "alpha"}}` are merged into the header too and
    dropped from the copy of `extra_body` that is sent (`main.py:193-196`), so the body has no `labels`."""
    with vertex(labelled.proxy, openai_echo, backend=MISTRAL) as rig:
        _chat(labelled, rig, extra_body={"labels": {"project": "alpha"}})
        call: Final = rig.peer.one_call()
        assert "labels" not in body_of(call)
        labels: Final = labels_header(call)
        assert labels is not None
        assert labels.get("project") == "alpha"


def test_a_caller_cannot_forge_the_reserved_identity_label(labelled: Labelled) -> None:
    """A caller label `user_api_key_alias: forged` is dropped because the `user_api_key_` prefix is
    reserved (`_is_reserved`), so the header still names the real alias and no label has the value
    `forged`."""
    with vertex(labelled.proxy, openai_echo, backend=MISTRAL) as rig:
        _chat(labelled, rig, labels={"user_api_key_alias": "forged", "team": "search"})
        labels: Final = labels_header(rig.peer.one_call())
        assert labels is not None
        assert labels["user_api_key_alias"] == _expected_alias(labelled.alias)
        assert "forged" not in labels.values()
        assert labels.get("team") == "search"


def test_mistral_labels_are_capped_at_thirty_two(labelled: Labelled) -> None:
    """With 40 caller labels the header holds exactly 32 labels (`VERTEX_PARTNER_REQUEST_LABELS_MAX`): the
    identity label first, then caller labels in order, then nothing from `spend_logs_metadata`."""
    many: Final[dict[str, JsonValue]] = {f"label{index:02d}": "v" for index in range(40)}
    with vertex(labelled.proxy, openai_echo, backend=MISTRAL) as rig:
        _chat(labelled, rig, labels=many)
        labels: Final = labels_header(rig.peer.one_call())
        assert labels is not None
        assert len(labels) == 32
        assert list(labels)[:3] == ["user_api_key_alias", "label00", "label01"]
        assert "cost_center" not in labels


def test_a_deployment_labels_header_stands_in_for_body_labels(labelled: Labelled) -> None:
    """When the deployment's `extra_headers` already carry a decodable `X-Vertex-AI-Labels`
    (`{"source": "deployment"}`), that header replaces the caller's body labels as the client labels
    (`vertex_partner_labels_header`), so the result has `source: deployment` and not the body's `source`."""
    header: Final = base64.b64encode(json.dumps({"source": "deployment"}).encode()).decode()
    with vertex(labelled.proxy, openai_echo, backend=MISTRAL, extra_headers={"X-Vertex-AI-Labels": header}) as rig:
        _chat(labelled, rig, labels={"source": "body"})
        labels: Final = labels_header(rig.peer.one_call())
        assert labels is not None
        assert labels.get("source") == "deployment"
        assert labels["user_api_key_alias"] == _expected_alias(labelled.alias)


def test_a_streamed_mistral_call_carries_the_labels_header(labelled: Labelled) -> None:
    """A streamed Mistral call (`:streamRawPredict`) carries the same identity labels header as a unary one."""
    with vertex(labelled.proxy, openai_echo, backend=MISTRAL) as rig:
        with labelled.proxy.client.stream(
            "POST",
            "/v1/chat/completions",
            json={"model": rig.model, "messages": [{"role": "user", "content": marker()}], "stream": True},
            headers={"Authorization": f"Bearer {labelled.key}"},
        ) as response:
            assert response.status_code == 200
            "".join(response.iter_text())
        labels: Final = labels_header(rig.peer.one_call())
        assert labels is not None
        assert labels["user_api_key_alias"] == _expected_alias(labelled.alias)


def test_codestral_chat_also_carries_the_labels_header(labelled: Labelled) -> None:
    """Codestral chat (not fill-in-the-middle) is in the Mistral partner family (`"codestral" in model`,
    `main.py:159`), so it gets the same identity labels header."""
    with vertex(labelled.proxy, openai_echo, backend=_CODESTRAL_CHAT) as rig:
        _chat(labelled, rig)
        labels: Final = labels_header(rig.peer.one_call())
        assert labels is not None
        assert labels["user_api_key_alias"] == _expected_alias(labelled.alias)


def test_a_key_without_alias_or_labels_sends_no_labels_header(labelled: Labelled) -> None:
    """A key with no alias and no `spend_logs_metadata`, sending no labels, resolves to no labels at all,
    so the Mistral call has no `X-Vertex-AI-Labels` header (`vertex_partner_labels_header` returns None
    for an empty result)."""
    with vertex(labelled.proxy, openai_echo, backend=MISTRAL) as rig:
        _chat(labelled, rig, key=labelled.bare_key)
        assert labels_header(rig.peer.one_call()) is None


def test_llama_maas_gets_no_labels_header_even_with_the_option_on(labelled: Labelled) -> None:
    """The labels header is computed only for the Mistral family (`main.py:178`), so a Llama MaaS call on
    the same proxy and key has no `X-Vertex-AI-Labels` header. A port that labels every partner model
    would change what Vertex bills these calls under."""
    with vertex(labelled.proxy, openai_echo, backend=LLAMA) as rig:
        _chat(labelled, rig)
        assert labels_header(rig.peer.one_call()) is None
