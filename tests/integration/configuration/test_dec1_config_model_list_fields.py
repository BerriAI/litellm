"""What each field of a `model_list` entry does once the config is loaded.

The contract per field: `model_name` is the public name callers ask for and `/v1/models` lists;
`litellm_params.model` is `<provider>/<upstream model>` and only the upstream part reaches the provider;
`api_base` and `api_key` say where and as whom; any other `litellm_params` entry is a default sent with
every call; `model_info.id` is the deployment's identity (derived deterministically when omitted) and is
echoed in the `x-litellm-model-id` response header; other `model_info` entries are metadata (pricing,
mode, access groups, free-form keys). Two entries with the same `model_name` form one load-balanced
group. Entries declared in the file are reported as `db_model: false` and never expose their `api_key`.

The Rust loader has to build identical deployments from the same YAML, including the derived ids, because
spend rows, health rows and cooldown keys are stored under those ids.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest
from pydantic import JsonValue

from tests.integration._support.client import (
    JSON_OBJECT,
    Gateway,
    eventually,
    gateway_from_environment,
    object_value,
    string_value,
)
from tests.integration._support.process import OwnedProxy
from tests.integration._support.wire import Wire
from tests.integration.configuration.dec1_config_support import (
    UPSTREAM_KEY,
    booted,
    chat,
    deployment,
    model_entries,
    refused,
    scripted_upstream,
    sent,
    sent_json,
    served_model_names,
)

_BASIC: Final = "dec1-basic"
_PAIR: Final = "dec1-pair"
_DEFAULTS: Final = "dec1-defaults"
_ORG: Final = "dec1-org"
_WILDCARD: Final = "dec1-any/*"
_PRICED_INFO: Final = "dec1-priced-by-model-info"
_PRICED_PARAMS: Final = "dec1-priced-by-litellm-params"
_GROUPED: Final = "dec1-grouped"
_NO_V1: Final = "dec1-base-without-v1"
_UNNAMED_ID: Final = "dec1-derived-id"
_HEX_ID: Final = re.compile(r"[0-9a-f]{64}")
# The scripted provider reports 10 prompt and 5 completion tokens on every reply.
_EXPECTED_COST: Final = 10 * 0.001 + 5 * 0.002


@dataclass(frozen=True, slots=True)
class _Rig:
    proxy: OwnedProxy
    upstream: Wire

    @property
    def gateway(self) -> Gateway:
        return self.proxy.gateway


def _model_list(upstream: Wire) -> list[dict[str, object]]:
    base: Final = f"{upstream.url}/v1"
    return [
        dict(
            deployment(
                _BASIC,
                base,
                model="openai/basic-upstream-model",
                model_info={"id": "dec1-fixed-id", "mode": "chat", "dec1_custom_field": "kept"},
                rpm=120,
                tpm=9000,
            )
        ),
        dict(deployment(_PAIR, base, model_info={"id": "dec1-pair-a"})),
        dict(deployment(_PAIR, base, model_info={"id": "dec1-pair-b"})),
        dict(deployment(_DEFAULTS, base, temperature=0.25, max_tokens=17)),
        dict(deployment(_ORG, base, organization="org-dec1-deployment")),
        dict(deployment(_WILDCARD, base, model="openai/*")),
        dict(
            deployment(
                _PRICED_INFO,
                base,
                model="openai/dec1-unpriced-model-a",
                model_info={"input_cost_per_token": 0.001, "output_cost_per_token": 0.002},
            )
        ),
        dict(
            deployment(
                _PRICED_PARAMS,
                base,
                model="openai/dec1-unpriced-model-b",
                input_cost_per_token=0.001,
                output_cost_per_token=0.002,
            )
        ),
        dict(deployment(_GROUPED, base, model_info={"access_groups": ["dec1-access-group"]})),
        dict(deployment(_NO_V1, upstream.url)),
        dict(deployment(_UNNAMED_ID, base)),
    ]


@pytest.fixture(scope="module")
def listed(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("dec1-model-list")
    with (
        scripted_upstream() as upstream,
        gateway_from_environment() as shared,
        booted(shared, directory, {"model_list": _model_list(upstream)}) as proxy,
    ):
        yield _Rig(proxy, upstream)


def _only(rig: _Rig, name: str) -> dict[str, JsonValue]:
    entries: Final = model_entries(rig.gateway, name)
    assert len(entries) == 1, entries
    return entries[0]


def test_model_name_is_the_public_name_and_only_the_upstream_part_of_model_reaches_the_provider(
    listed: _Rig,
) -> None:
    """Entry: `model_name: dec1-basic`, `litellm_params.model: openai/basic-upstream-model`. A chat
    completion asking for `dec1-basic` must reach the provider as `POST /v1/chat/completions` with
    `"model": "basic-upstream-model"`: neither the public name nor the `openai/` prefix leaves the proxy."""
    sent(listed.upstream)
    response: Final = chat(listed.gateway, _BASIC)
    assert response.status_code == 200, response.text
    requests: Final = sent(listed.upstream)
    assert [(request.method, request.target) for request in requests] == [("POST", "/v1/chat/completions")]
    assert sent_json(requests[0])["model"] == "basic-upstream-model"


def test_the_models_endpoint_lists_model_names_not_upstream_models(listed: _Rig) -> None:
    """`GET /v1/models` must list the configured `model_name` values and must not list the upstream model
    string of any deployment."""
    names: Final = served_model_names(listed.gateway)
    assert {_BASIC, _PAIR, _DEFAULTS} <= names, sorted(names)
    assert "basic-upstream-model" not in names and "openai/basic-upstream-model" not in names


def test_an_unconfigured_model_name_is_a_400_naming_the_model(listed: _Rig) -> None:
    """A chat completion for a name that is in no `model_list` entry must return 400 `invalid_request_error`
    whose message names the model (`model=dec1-not-configured`); the provider must not be called. The
    wording depends on which layer refuses: the route's `Invalid model name passed in model=...`
    (litellm/proxy/route_llm_request.py `ProxyModelNotFoundError`) or, when the name is handed to the router,
    `You passed in model=... There are no healthy deployments for this model`, so only the shared part is
    pinned."""
    sent(listed.upstream)
    response: Final = chat(listed.gateway, "dec1-not-configured")
    assert response.status_code == 400, response.text
    error: Final = object_value(response.json()["error"])
    assert error["type"] == "invalid_request_error", response.text
    assert "model=dec1-not-configured" in string_value(error["message"]), response.text
    assert sent(listed.upstream) == ()


def test_api_key_is_sent_to_the_provider_as_a_bearer_token(listed: _Rig) -> None:
    """`litellm_params.api_key` must arrive at the provider as `Authorization: Bearer <api_key>`; the
    caller's proxy key must never be forwarded in its place."""
    sent(listed.upstream)
    assert chat(listed.gateway, _BASIC).status_code == 200
    authorizations: Final = [request.headers["authorization"] for request in sent(listed.upstream)]
    assert authorizations == [f"Bearer {UPSTREAM_KEY}"]
    assert listed.gateway.key not in authorizations[0]


def test_api_base_without_a_version_segment_is_used_as_written(listed: _Rig) -> None:
    """`litellm_params.api_base` is the bare origin with no `/v1`. The provider must receive
    `POST /chat/completions`: the proxy appends the route to whatever base the config gives and does not
    add `/v1` itself."""
    sent(listed.upstream)
    response: Final = chat(listed.gateway, _NO_V1)
    assert response.status_code == 200, response.text
    assert [(request.method, request.target) for request in sent(listed.upstream)] == [("POST", "/chat/completions")]


def test_model_info_id_is_the_deployment_id_reported_in_the_response_header(listed: _Rig) -> None:
    """`model_info.id: dec1-fixed-id`. A chat completion served by that deployment must carry the response
    header `x-litellm-model-id: dec1-fixed-id`, and `/model/info` must report the same id."""
    response: Final = chat(listed.gateway, _BASIC)
    assert response.status_code == 200, response.text
    assert response.headers["x-litellm-model-id"] == "dec1-fixed-id"
    assert object_value(_only(listed, _BASIC)["model_info"])["id"] == "dec1-fixed-id"


def test_a_deployment_without_an_id_gets_a_64_hex_id(listed: _Rig) -> None:
    """An entry with no `model_info.id` must be given a derived id of 64 lowercase hex characters, and the
    `x-litellm-model-id` header of a call it serves must be that same id."""
    derived: Final = string_value(object_value(_only(listed, _UNNAMED_ID)["model_info"])["id"])
    assert _HEX_ID.fullmatch(derived), derived
    response: Final = chat(listed.gateway, _UNNAMED_ID)
    assert response.status_code == 200, response.text
    assert response.headers["x-litellm-model-id"] == derived


def test_a_derived_deployment_id_is_the_same_on_every_boot_of_the_same_config(gateway: Gateway, tmp_path: Path) -> None:
    """The same config file, with a deployment that has no `model_info.id`, is booted twice on fresh
    databases. `/model/info` must report the same derived id both times: the id is a pure function of the
    entry, so spend and health rows written before a restart (or by the Python loader before the Rust one
    takes over) still point at the same deployment."""
    config: Final = {"model_list": [deployment(_UNNAMED_ID, "http://127.0.0.1:9/v1")]}
    ids: Final[list[object]] = []
    for attempt in ("first", "second"):
        with booted(gateway, tmp_path / attempt, config) as owned:
            ids.append(object_value(model_entries(owned.gateway, _UNNAMED_ID)[0]["model_info"])["id"])
    assert ids[0] == ids[1], ids


def test_a_derived_deployment_id_changes_when_the_entry_changes(gateway: Gateway, tmp_path: Path) -> None:
    """Two entries that differ only in `api_base` and have no explicit id are loaded together. `/model/info`
    must report two different derived ids: the id covers the entry's parameters, not only its name."""
    config: Final = {
        "model_list": [
            deployment("dec1-twin-a", "http://127.0.0.1:9/v1"),
            deployment("dec1-twin-b", "http://127.0.0.1:9/v2"),
        ]
    }
    with booted(gateway, tmp_path, config) as owned:
        first: Final = object_value(model_entries(owned.gateway, "dec1-twin-a")[0]["model_info"])["id"]
        second: Final = object_value(model_entries(owned.gateway, "dec1-twin-b")[0]["model_info"])["id"]
    assert first != second


def test_file_declared_models_are_reported_as_not_database_models(listed: _Rig) -> None:
    """`/model/info` must report `model_info.db_model` as false for a deployment that came from the config
    file; the admin UI uses the flag to decide whether the entry can be edited."""
    assert object_value(_only(listed, _BASIC)["model_info"])["db_model"] is False


def test_model_info_never_returns_the_api_key(listed: _Rig) -> None:
    """`/model/info` must return the deployment's `litellm_params` without an `api_key` entry, and the key
    text must appear nowhere in the response body."""
    response: Final = listed.gateway.request("GET", "/model/info")
    assert response.status_code == 200, response.text
    assert "api_key" not in object_value(_only(listed, _BASIC)["litellm_params"])
    assert UPSTREAM_KEY not in response.text


def test_model_info_echoes_the_model_and_api_base_of_the_entry(listed: _Rig) -> None:
    """`/model/info` must return `litellm_params.model` exactly as written in the file
    (`openai/basic-upstream-model`) and the configured `api_base`."""
    params: Final = object_value(_only(listed, _BASIC)["litellm_params"])
    assert params["model"] == "openai/basic-upstream-model"
    assert params["api_base"] == f"{listed.upstream.url}/v1"


def test_rpm_and_tpm_limits_of_an_entry_are_echoed_as_numbers(listed: _Rig) -> None:
    """`litellm_params.rpm: 120` and `tpm: 9000`. `/model/info` must return both as the same integers
    (their names contain no secret, so they must not be masked)."""
    params: Final = object_value(_only(listed, _BASIC)["litellm_params"])
    assert (params["rpm"], params["tpm"]) == (120, 9000)


def test_model_info_mode_and_free_form_keys_are_echoed(listed: _Rig) -> None:
    """`model_info` holds `mode: chat` and an operator-defined `dec1_custom_field: kept`. `/model/info`
    must return both unchanged: unknown `model_info` keys are preserved, not dropped."""
    info: Final = object_value(_only(listed, _BASIC)["model_info"])
    assert info["mode"] == "chat"
    assert info["dec1_custom_field"] == "kept"


def test_two_entries_with_the_same_model_name_are_both_loaded_as_one_group(listed: _Rig) -> None:
    """Two entries share `model_name: dec1-pair`. `/model/info` must hold both deployments (ids
    `dec1-pair-a` and `dec1-pair-b`) while `/v1/models` lists the name once."""
    ids: Final = {object_value(entry["model_info"])["id"] for entry in model_entries(listed.gateway, _PAIR)}
    assert ids == {"dec1-pair-a", "dec1-pair-b"}
    listing: Final = listed.gateway.get("/v1/models")["data"]
    assert isinstance(listing, list)
    assert [object_value(item)["id"] for item in listing].count(_PAIR) == 1


def test_requests_to_a_duplicated_model_name_are_spread_over_both_deployments(listed: _Rig) -> None:
    """Repeated chat completions for `dec1-pair` must, within a bounded number of calls, be served by both
    deployments: the set of `x-litellm-model-id` response headers seen must become both ids."""
    seen: Final[set[str]] = set()

    def call() -> frozenset[str]:
        response: Final = chat(listed.gateway, _PAIR)
        assert response.status_code == 200, response.text
        seen.add(response.headers["x-litellm-model-id"])
        return frozenset(seen)

    assert eventually(call, lambda ids: len(ids) == 2, seconds=30) == {"dec1-pair-a", "dec1-pair-b"}


def test_extra_litellm_params_are_sent_as_defaults_on_every_call(listed: _Rig) -> None:
    """`litellm_params` holds `temperature: 0.25` and `max_tokens: 17`. A chat completion that names
    neither must reach the provider with both values in the JSON body."""
    sent(listed.upstream)
    assert chat(listed.gateway, _DEFAULTS).status_code == 200
    body: Final = sent_json(sent(listed.upstream)[0])
    assert (body["temperature"], body["max_tokens"]) == (0.25, 17)


def test_a_value_in_the_request_wins_over_the_deployment_default(listed: _Rig) -> None:
    """With `temperature: 0.25` as a deployment default, a chat completion that sends `temperature: 0.9`
    must reach the provider with 0.9.
    UNVERIFIED: request-over-deployment precedence for non-credential params; see how the router merges
    `litellm_params` with request kwargs."""
    sent(listed.upstream)
    assert chat(listed.gateway, _DEFAULTS, temperature=0.9).status_code == 200
    assert sent_json(sent(listed.upstream)[0])["temperature"] == 0.9


def test_organization_in_litellm_params_is_sent_as_the_openai_organization_header(listed: _Rig) -> None:
    """`litellm_params.organization: org-dec1-deployment`. The provider request must carry
    `openai-organization: org-dec1-deployment`."""
    sent(listed.upstream)
    assert chat(listed.gateway, _ORG).status_code == 200
    assert [request.headers.get("openai-organization") for request in sent(listed.upstream)] == ["org-dec1-deployment"]


def test_a_wildcard_entry_serves_any_name_under_its_prefix_with_the_requested_suffix(listed: _Rig) -> None:
    """Entry: `model_name: dec1-any/*`, `litellm_params.model: openai/*`. A chat completion for
    `dec1-any/some-new-model` must succeed and reach the provider with `"model": "some-new-model"`."""
    sent(listed.upstream)
    response: Final = chat(listed.gateway, "dec1-any/some-new-model")
    assert response.status_code == 200, response.text
    assert [sent_json(request)["model"] for request in sent(listed.upstream)] == ["some-new-model"]


def test_pricing_in_model_info_sets_the_response_cost_header(listed: _Rig) -> None:
    """`model_info` prices an otherwise unknown model at 0.001 per input token and 0.002 per output
    token. The provider reports 10 prompt and 5 completion tokens, so the response header
    `x-litellm-response-cost` must be 10 * 0.001 + 5 * 0.002 = 0.02."""
    response: Final = chat(listed.gateway, _PRICED_INFO)
    assert response.status_code == 200, response.text
    assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(_EXPECTED_COST)


def test_pricing_in_litellm_params_sets_the_response_cost_header(listed: _Rig) -> None:
    """The same prices written under `litellm_params` instead of `model_info` must yield the same
    `x-litellm-response-cost` of 0.02 for a 10 prompt / 5 completion token reply."""
    response: Final = chat(listed.gateway, _PRICED_PARAMS)
    assert response.status_code == 200, response.text
    assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(_EXPECTED_COST)


def test_model_info_access_groups_let_a_key_scoped_to_the_group_call_the_model(listed: _Rig) -> None:
    """`model_info.access_groups: [dec1-access-group]`. A virtual key whose `models` is that group name
    must be able to call the grouped model (200) and must be denied another model with 403 and error type
    `key_model_access_denied`.
    UNVERIFIED: the 403 for this exact path; other model-access denials in `auth_checks.py` use 403."""
    with listed.gateway.scenario() as scenario:
        key: Final = scenario.key(models=["dec1-access-group"])
        allowed: Final = chat(listed.gateway, _GROUPED, key=key)
        denied: Final = chat(listed.gateway, _BASIC, key=key)
    assert allowed.status_code == 200, allowed.text
    assert denied.status_code == 403, denied.text
    assert object_value(JSON_OBJECT.validate_json(denied.content)["error"])["type"] == "key_model_access_denied"


def test_an_entry_without_litellm_params_refuses_to_boot(gateway: Gateway, tmp_path: Path) -> None:
    """A `model_list` entry has a `model_name` and nothing else. The proxy must exit non-zero rather than
    serve a model it cannot call, and the log must mention `litellm_params`.
    UNVERIFIED: the exact error text (today a Python KeyError from the loader loop)."""
    log: Final = refused(gateway, tmp_path, {"model_list": [{"model_name": "dec1-incomplete"}]})
    assert "litellm_params" in log


def test_an_entry_whose_litellm_params_has_no_model_refuses_to_boot(gateway: Gateway, tmp_path: Path) -> None:
    """A `model_list` entry has `litellm_params` with an `api_base` but no `model`. The proxy must exit
    non-zero, and the log must show the missing `model` key.
    UNVERIFIED: the exact error text (today `KeyError: 'model'` from the loader loop)."""
    log: Final = refused(
        gateway,
        tmp_path,
        {"model_list": [{"model_name": "dec1-incomplete", "litellm_params": {"api_base": "http://127.0.0.1:9"}}]},
    )
    assert "KeyError: 'model'" in log


def test_an_empty_model_list_boots_and_lists_no_models(gateway: Gateway, tmp_path: Path) -> None:
    """`model_list: []`. The proxy must boot, answer `GET /v1/models` with 200 and an empty `data` list,
    and answer a chat completion with 400 `Invalid model name passed in model=...`.
    UNVERIFIED: that an empty list takes the same not-found path as an unknown name in a non-empty list."""
    with booted(gateway, tmp_path, {"model_list": []}) as owned:
        listing: Final = owned.gateway.get("/v1/models")
        response: Final = chat(owned.gateway, "dec1-anything")
    assert listing["data"] == []
    assert response.status_code == 400, response.text
    assert "Invalid model name passed in model=dec1-anything" in response.text
