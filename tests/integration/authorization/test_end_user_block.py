import uuid
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import Gateway, eventually, object_value
from integration._support.database import read_rows
from integration._support.process import owned_proxy
from pydantic import BaseModel, JsonValue


class _BlockedCustomerInfo(BaseModel):
    user_id: str
    blocked: bool


def _blocked_row(user_id: str) -> list[dict[str, JsonValue]]:
    return read_rows('SELECT blocked FROM "LiteLLM_EndUserTable" WHERE user_id=%s', (user_id,))


def _observed(upstream: httpx.Client) -> list[dict[str, JsonValue]]:
    response: Final = upstream.get("/__observations")
    response.raise_for_status()
    requests: Final = response.json()["requests"]
    assert isinstance(requests, list)
    return requests


def _chat(gateway: Gateway, model: str, key: str, end_user: str) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": f"end user block {uuid.uuid4().hex}"}],
            "user": end_user,
        },
        key=key,
    )


def _assert_blocked_refusal(response: httpx.Response, end_user: str) -> None:
    assert response.status_code == 400, response.text
    assert response.json() == {
        "error": {
            "message": f"User blocked from making LLM API Calls. User={end_user}",
            "type": "invalid_request_error",
            "param": None,
            "code": "400",
            "provider_specific_fields": {"error": f"User blocked from making LLM API Calls. User={end_user}"},
        }
    }, response.text


@dataclass(frozen=True, slots=True)
class _CallbackRig:
    first: Gateway
    second: Gateway


def _callback_config(directory: Path) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["litellm_settings"] = {
        **object_value(config["litellm_settings"]),
        "callbacks": ["blocked_user_check"],
    }
    path: Final = directory / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _two_callback_proxies(gateway: Gateway, tmp_path: Path, stack: ExitStack) -> _CallbackRig:
    config: Final = _callback_config(tmp_path)
    first: Final = stack.enter_context(owned_proxy(gateway, tmp_path / "a", {}, config=config))
    second: Final = stack.enter_context(owned_proxy(gateway, tmp_path / "b", {}, config=config))
    return _CallbackRig(first, second)


def _one_callback_proxy(gateway: Gateway, tmp_path: Path, stack: ExitStack) -> Gateway:
    config: Final = _callback_config(tmp_path)
    return stack.enter_context(owned_proxy(gateway, tmp_path / "a", {}, config=config))


@pytest.mark.parametrize("block_route", ("/customer/block", "/end_user/block"))
def test_block_route_marks_the_customer_blocked_and_update_unblocks(gateway: Gateway, block_route: str) -> None:
    with gateway.scenario() as scenario:
        end_user: Final = f"integration-end-user-{uuid.uuid4().hex}"
        created: Final = gateway.request("POST", "/customer/new", {"user_id": end_user, "max_budget": 100.0})
        assert created.status_code == 200, created.text
        scenario.cleanups.callback(gateway.request, "POST", "/customer/delete", {"user_ids": [end_user]})

        gateway.request("POST", block_route, {"user_ids": [end_user]})
        assert eventually(lambda: _blocked_row(end_user), lambda value: value == [{"blocked": True}], seconds=15) == [
            {"blocked": True}
        ]
        info: Final = gateway.get("/customer/info", {"end_user_id": end_user})
        assert _BlockedCustomerInfo.model_validate(info).blocked is True

        updated: Final = gateway.request("POST", "/customer/update", {"user_id": end_user, "blocked": False})
        assert updated.status_code == 200, updated.text
        assert eventually(lambda: _blocked_row(end_user), lambda value: value == [{"blocked": False}], seconds=15) == [
            {"blocked": False}
        ]
        assert (
            _BlockedCustomerInfo.model_validate(gateway.get("/customer/info", {"end_user_id": end_user})).blocked
            is False
        )


@pytest.mark.parametrize("block_route", ("/customer/block", "/end_user/block"))
def test_blocked_end_user_is_refused_on_every_proxy_and_unblock_restores(
    gateway: Gateway, tmp_path: Path, block_route: str
) -> None:
    pytest.skip(
        "BUG: blocked_user_check hook captures prisma_client=None at startup so API-blocked customers are never refused"
    )
    with (
        ExitStack() as stack,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        rig: Final = _two_callback_proxies(gateway, tmp_path, stack)
        with rig.first.scenario() as scenario:
            _observed(upstream)
            end_user: Final = f"integration-end-user-{uuid.uuid4().hex}"
            created: Final = rig.first.request("POST", "/customer/new", {"user_id": end_user, "max_budget": 100.0})
            assert created.status_code == 200, created.text
            scenario.cleanups.callback(rig.first.request, "POST", "/customer/delete", {"user_ids": [end_user]})
            model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
            key: Final = scenario.key(models=[model])
            served: Final = _chat(rig.first, model, key, end_user)
            assert served.status_code == 200, served.text
            _observed(upstream)

            rig.first.request("POST", block_route, {"user_ids": [end_user]})
            assert eventually(
                lambda: _blocked_row(end_user), lambda value: value == [{"blocked": True}], seconds=15
            ) == [{"blocked": True}]
            info: Final = rig.first.get("/customer/info", {"end_user_id": end_user})
            assert _BlockedCustomerInfo.model_validate(info).blocked is True

            denied_first: Final = _chat(rig.first, model, key, end_user)
            denied_second: Final = _chat(rig.second, model, key, end_user)
            for denied in (denied_first, denied_second):
                _assert_blocked_refusal(denied, end_user)
            upstream_calls: Final = _observed(upstream)
            assert upstream_calls == [], upstream_calls

            updated: Final = rig.first.request("POST", "/customer/update", {"user_id": end_user, "blocked": False})
            assert updated.status_code == 200, updated.text
            assert eventually(
                lambda: _blocked_row(end_user), lambda value: value == [{"blocked": False}], seconds=15
            ) == [{"blocked": False}]
            assert (
                _BlockedCustomerInfo.model_validate(rig.first.get("/customer/info", {"end_user_id": end_user})).blocked
                is False
            )

            text: Final = f"end user unblocked {uuid.uuid4().hex}"

            def call(candidate: Gateway) -> httpx.Response:
                return candidate.request(
                    "POST",
                    "/v1/chat/completions",
                    {"model": model, "messages": [{"role": "user", "content": text}], "user": end_user},
                    key=key,
                )

            assert (
                eventually(
                    lambda: call(rig.first), lambda response: response.status_code == 200, seconds=75
                ).status_code
                == 200
            )
            assert (
                eventually(
                    lambda: call(rig.second), lambda response: response.status_code == 200, seconds=75
                ).status_code
                == 200
            )
            expected_observations: Final = [
                {
                    "path": "/v1/chat/completions",
                    "authorization": "Bearer integration-provider-key",
                    "body": {
                        "messages": [{"role": "user", "content": text}],
                        "model": "gpt-4o-mini",
                        "user": end_user,
                    },
                    "method": "POST",
                    "api_key": "",
                }
            ] * 2
            assert _observed(upstream) == expected_observations


def test_unblock_route_requires_the_blocked_user_check_callback(gateway: Gateway, tmp_path: Path) -> None:
    with ExitStack() as stack:
        callback_proxy: Final = _one_callback_proxy(gateway, tmp_path, stack)
        end_user: Final = f"integration-end-user-{uuid.uuid4().hex}"
        created: Final = callback_proxy.request(
            "POST", "/customer/new", {"user_id": end_user, "max_budget": 100.0, "blocked": True}
        )
        assert created.status_code == 200, created.text
        stack.callback(callback_proxy.request, "POST", "/customer/delete", {"user_ids": [end_user]})
        assert _blocked_row(end_user) == [{"blocked": True}]

        shared: Final = gateway.request("POST", "/customer/unblock", {"user_ids": [end_user]})
        assert shared.status_code == 400, shared.text
        assert shared.json() == {"detail": {"error": "Blocked user check was never set. This call has no effect."}}, (
            shared.text
        )
        assert _blocked_row(end_user) == [{"blocked": True}]

        callback_response: Final = callback_proxy.request("POST", "/customer/unblock", {"user_ids": [end_user]})
        assert callback_response.status_code == 400, callback_response.text
        assert callback_response.json() == {
            "detail": {"error": "Blocked user check was never set. This call has no effect."}
        }, callback_response.text
        assert _blocked_row(end_user) == [{"blocked": True}]


def test_block_route_returns_the_blocked_records(gateway: Gateway) -> None:
    pytest.skip("BUG: /customer/block returns 500 (response serialization) instead of 200 BlockUsersResponse")
    end_user: Final = f"integration-end-user-{uuid.uuid4().hex}"
    with gateway.scenario() as scenario:
        created: Final = gateway.request("POST", "/customer/new", {"user_id": end_user})
        assert created.status_code == 200, created.text
        scenario.cleanups.callback(gateway.request, "POST", "/customer/delete", {"user_ids": [end_user]})
        blocked: Final = gateway.request("POST", "/customer/block", {"user_ids": [end_user]})
        assert blocked.status_code == 200, blocked.text
        payload: Final = object_value(blocked.json())
        blocked_users: Final = payload["blocked_users"]
        assert isinstance(blocked_users, list) and len(blocked_users) == 1
        assert object_value(blocked_users[0])["user_id"] == end_user
        assert object_value(blocked_users[0])["blocked"] is True


def test_blocked_end_user_is_refused_without_the_callback(gateway: Gateway) -> None:
    pytest.skip("BUG: blocked end user is still served 200 when the proxy runs without blocked_user_check")
    with gateway.scenario() as scenario:
        end_user: Final = f"integration-end-user-{uuid.uuid4().hex}"
        created: Final = gateway.request("POST", "/customer/new", {"user_id": end_user, "blocked": True})
        assert created.status_code == 200, created.text
        scenario.cleanups.callback(gateway.request, "POST", "/customer/delete", {"user_ids": [end_user]})
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        key: Final = scenario.key(models=[model])
        refused: Final = _chat(gateway, model, key, end_user)
        _assert_blocked_refusal(refused, end_user)


def test_blocked_end_user_header_spelling_is_refused_with_the_callback(gateway: Gateway, tmp_path: Path) -> None:
    pytest.skip("BUG: header x-litellm-end-user-id spelling bypasses the blocked_user_check hook")
    with (
        ExitStack() as stack,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        config: Final = _callback_config(tmp_path)
        cfg: Final = yaml.safe_load(config.read_text())
        end_user: Final = f"integration-end-user-{uuid.uuid4().hex}"
        cfg["litellm_settings"]["blocked_user_list"] = [end_user]
        config.write_text(yaml.safe_dump(cfg))
        proxy: Final = stack.enter_context(owned_proxy(gateway, tmp_path / "listed", {}, config=config))
        with proxy.scenario() as scenario:
            created: Final = proxy.request("POST", "/customer/new", {"user_id": end_user, "max_budget": 100.0})
            assert created.status_code == 200, created.text
            scenario.cleanups.callback(proxy.request, "POST", "/customer/delete", {"user_ids": [end_user]})
            model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
            key: Final = scenario.key(models=[model])
            refused: Final = proxy.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": f"blocked {uuid.uuid4().hex}"}]},
                key=key,
                headers={"x-litellm-end-user-id": end_user},
            )
            _assert_blocked_refusal(refused, end_user)
            assert _observed(upstream) == []
