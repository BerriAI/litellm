import json
import uuid
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import Gateway, JsonValue, gateway_from_environment
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server

OPENAI_MODEL: Final = "gpt-4.1-mini"
MODEL_TPM: Final = 1000
SHARE_TPM: Final = 100
TOKENS_PER_CALL: Final = 30
MAX_BORROW_MULTIPLE: Final = 2
ORG_ALIAS_HEADER: Final = "x-litellm-customer-org-alias"
UPSTREAM_REPLY: Final = json.dumps(
    {
        "id": "chatcmpl_customer_org_and_key_borrowing",
        "object": "chat.completion",
        "created": 1700000000,
        "model": OPENAI_MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "borrowed"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": TOKENS_PER_CALL},
    }
).encode()


def _respond(request: Request) -> Reply:
    assert request.method == "POST" and request.target.endswith("/v1/chat/completions"), request.target
    return Reply(body=UPSTREAM_REPLY)


def _proxy(
    tmp_path_factory: pytest.TempPathFactory, saturation_threshold: float, max_borrow_multiple: int | None = None
) -> Iterator[tuple[Gateway, str]]:
    directory: Final = tmp_path_factory.mktemp(f"org-key-borrowing-{saturation_threshold}-{max_borrow_multiple}")
    configuration: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    configuration["litellm_settings"] = {
        **configuration["litellm_settings"],
        "callbacks": ["dynamic_rate_limiter_v3"],
        "priority_reservation_settings": {
            "saturation_threshold": saturation_threshold,
            "model_limit_borrowing_config": {
                "borrow_idle_capacity": ["key", "customer_organization"],
                "max_borrow_multiple": max_borrow_multiple,
            },
        },
    }
    path: Final = directory / "org_key_model_limit_borrowing.yaml"
    path.write_text(yaml.safe_dump(configuration))
    with (
        gateway_from_environment() as rig,
        wire_server(_respond) as wire,
        owned_proxy(rig, directory, {}, config=path) as owned,
    ):
        yield owned, wire.url


@pytest.fixture(scope="module")
def idle_model(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Gateway, str]]:
    yield from _proxy(tmp_path_factory, saturation_threshold=0.9)


@pytest.fixture(scope="module")
def idle_model_with_a_borrow_cap(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Gateway, str]]:
    yield from _proxy(tmp_path_factory, saturation_threshold=0.9, max_borrow_multiple=MAX_BORROW_MULTIPLE)


@pytest.fixture(scope="module")
def saturated_model(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Gateway, str]]:
    yield from _proxy(tmp_path_factory, saturation_threshold=0.0)


def _customer_org(owned: Gateway, budget: Mapping[str, JsonValue]) -> str:
    alias: Final = f"borrowing-org-{uuid.uuid4().hex[:8]}"
    created: Final = owned.request("POST", "/management/v1/customer_organizations", {"alias": alias, "budget": budget})
    assert created.status_code == 201, created.text
    return alias


def _chat(owned: Gateway, model: str, key: str, org_alias: str | None = None) -> httpx.Response:
    return owned.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": f"org and key borrowing {uuid.uuid4().hex}"}]},
        key=key,
        headers={} if org_alias is None else {ORG_ALIAS_HEADER: org_alias},
    )


def _org_calls(owned: Gateway, upstream: str, calls: int) -> tuple[httpx.Response, ...]:
    """One customer organization calling through two service keys in turn."""
    with owned.scenario() as scenario:
        model: Final = scenario.model(
            model=f"openai/{OPENAI_MODEL}", api_base=f"{upstream}/v1", api_key="synthetic-openai-key", tpm=MODEL_TPM
        )
        org: Final = _customer_org(owned, {"model_tpm_limit": {model: SHARE_TPM}})
        keys: Final = (scenario.key(), scenario.key())
        return tuple(_chat(owned, model, keys[call % 2], org) for call in range(calls))


def _key_calls(
    owned: Gateway, upstream: str, calls: int, max_borrow_multiple: int | None = None
) -> tuple[httpx.Response, ...]:
    with owned.scenario() as scenario:
        model: Final = scenario.model(
            model=f"openai/{OPENAI_MODEL}", api_base=f"{upstream}/v1", api_key="synthetic-openai-key", tpm=MODEL_TPM
        )
        metadata: Final[Mapping[str, JsonValue]] = (
            {} if max_borrow_multiple is None else {"model_limit_max_borrow_multiple": max_borrow_multiple}
        )
        key: Final = scenario.key(model_tpm_limit={model: SHARE_TPM}, metadata=dict(metadata))
        return tuple(_chat(owned, model, key) for _ in range(calls))


def test_a_customer_org_borrows_idle_model_capacity_past_its_model_tpm_limit_across_its_keys(
    idle_model: tuple[Gateway, str],
) -> None:
    owned, upstream = idle_model
    responses: Final = _org_calls(owned, upstream, calls=8)
    assert [response.status_code for response in responses] == [200] * 8, [r.text for r in responses]


def test_a_customer_org_is_held_to_its_model_tpm_limit_once_the_model_saturates(
    saturated_model: tuple[Gateway, str],
) -> None:
    owned, upstream = saturated_model
    responses: Final = _org_calls(owned, upstream, calls=8)
    blocked: Final = tuple(response for response in responses if response.status_code == 429)
    assert responses[0].status_code == 200, responses[0].text
    assert blocked, [r.text for r in responses]
    assert blocked[0].headers.get("rate_limited_by") == "model_per_customer_org_saturated", dict(blocked[0].headers)
    assert "Customer organization model_tpm_limit reached" in blocked[0].json()["error"]["message"], blocked[0].text
    admitted: Final = sum(1 for response in responses if response.status_code == 200)
    assert admitted * TOKENS_PER_CALL <= SHARE_TPM + TOKENS_PER_CALL, [r.status_code for r in responses]


def test_a_customer_org_stops_borrowing_at_the_configured_max_borrow_multiple_while_the_model_is_idle(
    idle_model_with_a_borrow_cap: tuple[Gateway, str],
) -> None:
    owned, upstream = idle_model_with_a_borrow_cap
    responses: Final = _org_calls(owned, upstream, calls=12)
    blocked: Final = tuple(response for response in responses if response.status_code == 429)
    admitted: Final = sum(1 for response in responses if response.status_code == 200)
    assert blocked, [r.text for r in responses]
    assert blocked[0].headers.get("rate_limited_by") == "model_per_customer_org_max_borrow_multiple", dict(
        blocked[0].headers
    )
    assert SHARE_TPM < admitted * TOKENS_PER_CALL <= SHARE_TPM * MAX_BORROW_MULTIPLE + TOKENS_PER_CALL


def test_a_key_borrows_idle_model_capacity_past_its_own_model_tpm_limit(idle_model: tuple[Gateway, str]) -> None:
    owned, upstream = idle_model
    responses: Final = _key_calls(owned, upstream, calls=8)
    assert [response.status_code for response in responses] == [200] * 8, [r.text for r in responses]


def test_a_key_is_held_to_its_own_model_tpm_limit_once_the_model_saturates(
    saturated_model: tuple[Gateway, str],
) -> None:
    owned, upstream = saturated_model
    responses: Final = _key_calls(owned, upstream, calls=8)
    blocked: Final = tuple(response for response in responses if response.status_code == 429)
    assert responses[0].status_code == 200, responses[0].text
    assert blocked, [r.text for r in responses]
    assert blocked[0].headers.get("rate_limited_by") == "model_per_key_saturated", dict(blocked[0].headers)
    assert "Key model_tpm_limit reached" in blocked[0].json()["error"]["message"], blocked[0].text


def test_a_key_stops_borrowing_at_its_max_borrow_multiple_while_the_model_is_idle(
    idle_model: tuple[Gateway, str],
) -> None:
    owned, upstream = idle_model
    responses: Final = _key_calls(owned, upstream, calls=12, max_borrow_multiple=MAX_BORROW_MULTIPLE)
    blocked: Final = tuple(response for response in responses if response.status_code == 429)
    admitted: Final = sum(1 for response in responses if response.status_code == 200)
    assert blocked, [r.text for r in responses]
    assert blocked[0].headers.get("rate_limited_by") == "model_per_key_max_borrow_multiple", dict(blocked[0].headers)
    assert SHARE_TPM < admitted * TOKENS_PER_CALL <= SHARE_TPM * MAX_BORROW_MULTIPLE + TOKENS_PER_CALL
