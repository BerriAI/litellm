"""Contract: `/v1/containers` (OpenAI code interpreter containers) through an `openai/` deployment.

`POST /v1/containers` with `model: <alias>` in the body is routed to that deployment (`container_endpoints/
endpoints.py:32`); the provider body is `{"name", "expires_after"?, "file_ids"?}` only
(`ContainerRequestUtils.get_requested_container_create_optional_param`, `litellm/containers/utils.py:42`), and the
id the caller receives is a managed `cntr_` id that wraps provider, deployment id and raw id. The proxy then records
the container as owned by the creating key (`ownership.py:186`):

- retrieve and delete decode the managed id, send the raw id to the same deployment and are refused with 403
  `Forbidden` for any other non-admin key (`assert_user_can_access_container`, `ownership.py:328`);
- a non-admin list is filtered to the containers the key owns (`list_owned_containers`), an admin list is the
  provider's page;
- each create is priced as one code interpreter session (`openai/container` row, 0.03) and the price rides on the
  `llm_provider-x-litellm-response-cost` response header.

The container file routes are pinned in `test_container_file_routes_wire.py` and not repeated.
"""

from __future__ import annotations

from typing import Final

import pytest
from integration._support.client import JSON_OBJECT, Gateway, list_value, object_value, string_value
from integration._support.wire import Reply
from integration.providers.dec1_openai2_support import (
    CONTAINER_SESSION_COST,
    calls_to,
    container_object,
    decoded_managed_id,
    deployment_id,
    resource_respond,
)
from integration.providers.dec1_openai_support import PROVIDER_KEY, Call, deployment, json_reply, peer
from pydantic import JsonValue

_MODEL: Final = "openai/gpt-4o-mini"
_RAW: Final = "cntr_dec1openai2raw"


def _create(gateway: Gateway, model: str, key: str | None = None, **fields: JsonValue) -> str:
    response: Final = gateway.request(
        "POST", "/v1/containers", {"model": model, "name": "dec1-openai2", **fields}, key=key
    )
    assert response.status_code == 200, response.text
    return string_value(JSON_OBJECT.validate_json(response.content)["id"])


def test_create_forwards_name_expiry_and_file_ids_and_nothing_else(gateway: Gateway) -> None:
    """The caller creates a container with `name`, `expires_after: {"anchor": "last_active_at", "minutes": 20}`
    and `file_ids: ["file-dec1-a"]` on a model alias.

    The provider receives one `POST /v1/containers` with the deployment key and a body of exactly those three
    fields: `model` (the proxy routing hint) is not forwarded.
    """
    with peer(resource_respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        _create(
            gateway,
            model,
            expires_after={"anchor": "last_active_at", "minutes": 20},
            file_ids=["file-dec1-a"],
        )
        (call,) = calls_to(provider.posts(), "/containers")
        assert call.path == "/v1/containers"
        assert call.headers["authorization"] == f"Bearer {PROVIDER_KEY}"
        assert call.body == {
            "name": "dec1-openai2",
            "expires_after": {"anchor": "last_active_at", "minutes": 20},
            "file_ids": ["file-dec1-a"],
        }


def test_create_returns_a_managed_id_naming_the_provider_deployment_and_raw_id(gateway: Gateway) -> None:
    """The provider answers id `cntr_dec1openai2raw`.

    The caller's id is `cntr_` plus base64 of
    `litellm:custom_llm_provider:openai;model_id:<deployment id>;container_id:cntr_dec1openai2raw`, so later calls
    with only the id find the deployment again.
    """
    with peer(resource_respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        managed: Final = _create(gateway, model)
        assert decoded_managed_id(managed, "cntr_") == (
            f"litellm:custom_llm_provider:openai;model_id:{deployment_id(gateway, model)};container_id:{_RAW}"
        )


def test_create_forwards_memory_limit(gateway: Gateway) -> None:
    """The caller creates a container with `memory_limit: "4g"`, an OpenAI create field.

    The provider body must carry `memory_limit: "4g"`, otherwise the container gets the default size silently.
    """
    pytest.skip(
        "BUG: get_requested_container_create_optional_param keeps only expires_after, file_ids, extra_headers and "
        "extra_body, so memory_limit is dropped before the provider"
    )
    with peer(resource_respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        _create(gateway, model, memory_limit="4g")
        (call,) = calls_to(provider.posts(), "/containers")
        assert call.body["memory_limit"] == "4g"


def test_retrieve_by_the_managed_id_reaches_the_raw_container(gateway: Gateway) -> None:
    """The creating key calls `GET /v1/containers/<managed id>`.

    The provider receives `GET /v1/containers/cntr_dec1openai2raw` with the deployment key and the caller gets
    200 with `object: container` and `status: running`.
    """
    with peer(resource_respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        owner: Final = scenario.key(models=[model])
        managed: Final = _create(gateway, model, key=owner)
        response: Final = gateway.request("GET", f"/v1/containers/{managed}", key=owner)
        assert response.status_code == 200, response.text
        body: Final = JSON_OBJECT.validate_json(response.content)
        assert (body["object"], body["status"]) == ("container", "running"), body
        gets: Final = [call for call in calls_to(provider.all(), "/containers/") if call.method == "GET"]
        assert [call.path for call in gets] == [f"/v1/containers/{_RAW}"], gets
        assert gets[0].headers["authorization"] == f"Bearer {PROVIDER_KEY}"


def test_delete_by_the_managed_id_deletes_the_raw_container(gateway: Gateway) -> None:
    """The creating key calls `DELETE /v1/containers/<managed id>`.

    The provider receives `DELETE /v1/containers/cntr_dec1openai2raw` and the caller gets 200 with
    `deleted: true`.
    """
    with peer(resource_respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        owner: Final = scenario.key(models=[model])
        managed: Final = _create(gateway, model, key=owner)
        response: Final = gateway.request("DELETE", f"/v1/containers/{managed}", key=owner)
        assert response.status_code == 200, response.text
        assert JSON_OBJECT.validate_json(response.content)["deleted"] is True
        deletes: Final = [call.path for call in provider.all() if call.method == "DELETE"]
        assert deletes == [f"/v1/containers/{_RAW}"], deletes


@pytest.mark.parametrize("method", ["GET", "DELETE"])
def test_another_key_is_refused_with_403_before_the_provider(gateway: Gateway, method: str) -> None:
    """Key A creates a container; key B, allowed on the same model, calls `<method> /v1/containers/<id>`.

    Key B gets 403 with `Forbidden` and the provider sees no request for the container, so one tenant cannot
    read or delete another tenant's code interpreter sandbox by guessing its id.
    """
    with peer(resource_respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        owner: Final = scenario.key(models=[model])
        stranger: Final = scenario.key(models=[model])
        managed: Final = _create(gateway, model, key=owner)
        response: Final = gateway.request(method, f"/v1/containers/{managed}", key=stranger)
        assert response.status_code == 403, response.text
        assert "Forbidden" in response.text
        assert not calls_to(provider.all(), "/containers/"), provider.all()


def test_a_raw_provider_id_without_an_ownership_row_is_admin_only(gateway: Gateway) -> None:
    """A non-admin key calls `GET /v1/containers/cntr_dec1untracked`, an id the proxy never handed out.

    The key gets 403: untracked containers have no owner and only an admin may reach them
    (`ownership.py:338`). The provider sees nothing.
    """
    with peer(resource_respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        key: Final = scenario.key(models=[model])
        response: Final = gateway.request("GET", "/v1/containers/cntr_dec1untracked", key=key, params={"model": model})
        assert response.status_code == 403, response.text
        assert not calls_to(provider.all(), "/containers/"), provider.all()


def test_admin_list_forwards_limit_order_and_after(gateway: Gateway) -> None:
    """The master key calls `GET /v1/containers?model=<alias>&limit=5&order=asc&after=cntr_dec1prev`.

    The provider receives one `GET /v1/containers` whose query has `limit=5`, `order=asc` and
    `after=cntr_dec1prev`, and the admin gets the provider page unfiltered (`data[0].id` is the listed container).
    UNVERIFIED: the routing `model` query value is not forwarded to the provider and the listed id is relayed as is.
    """
    with peer(resource_respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        response: Final = gateway.request(
            "GET",
            "/v1/containers",
            params={"model": model, "limit": "5", "order": "asc", "after": "cntr_dec1prev"},
        )
        assert response.status_code == 200, response.text
        (call,) = [c for c in calls_to(provider.all(), "/containers") if c.method == "GET"]
        assert (call.query["limit"], call.query["order"], call.query["after"]) == (["5"], ["asc"], ["cntr_dec1prev"])
        assert "model" not in call.query, call.query
        data: Final = list_value(JSON_OBJECT.validate_json(response.content)["data"])
        assert object_value(data[0])["id"] == "cntr_dec1openai2listed", data


def test_a_key_lists_only_the_containers_it_created(gateway: Gateway) -> None:
    """Key A creates a container (raw `cntr_dec1openai2raw`); the provider's list page holds that container and
    `cntr_dec1foreign`, which nobody created through the proxy.

    `GET /v1/containers?model=<alias>` with key A returns only A's container: the foreign one is filtered out
    by the ownership table. UNVERIFIED: the kept item is matched on its raw id and relayed with the provider's id.
    """

    def respond(call: Call) -> Reply:
        if call.method == "GET" and call.path == "/v1/containers":
            return json_reply(
                {
                    "object": "list",
                    "data": [container_object(_RAW), container_object("cntr_dec1foreign")],
                    "first_id": _RAW,
                    "last_id": "cntr_dec1foreign",
                    "has_more": False,
                }
            )
        return resource_respond(call)

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        owner: Final = scenario.key(models=[model])
        _create(gateway, model, key=owner)
        response: Final = gateway.request("GET", "/v1/containers", key=owner, params={"model": model})
        assert response.status_code == 200, response.text
        ids: Final = [
            object_value(item)["id"] for item in list_value(JSON_OBJECT.validate_json(response.content)["data"])
        ]
        assert "cntr_dec1foreign" not in ids and len(ids) == 1, ids


def test_create_reports_one_code_interpreter_session_as_its_cost(gateway: Gateway) -> None:
    """A container create on an OpenAI deployment is answered by the provider.

    The caller's response carries `llm_provider-x-litellm-response-cost: 0.03`, one session at the
    `openai/container` row's `code_interpreter_cost_per_session` (`containers/transformation.py:167`).
    UNVERIFIED: the hidden `additional_headers` entry is surfaced as a response header with that exact name.
    """
    with peer(resource_respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        response: Final = gateway.request("POST", "/v1/containers", {"model": model, "name": "dec1-cost"})
        assert response.status_code == 200, response.text
        assert float(response.headers["llm_provider-x-litellm-response-cost"]) == pytest.approx(CONTAINER_SESSION_COST)


def test_a_provider_error_on_create_keeps_its_status_and_records_no_owner(gateway: Gateway) -> None:
    """The provider answers the create with 400 `dec1 bad expiry`.

    The caller gets 400 with that message and no container id; a follow-up list by the same key shows nothing
    owned, because ownership is only recorded after a successful create.
    """

    def respond(call: Call) -> Reply:
        if call.method == "POST" and call.path == "/v1/containers":
            return json_reply({"error": {"message": "dec1 bad expiry", "type": "invalid_request_error"}}, 400)
        return resource_respond(call)

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _MODEL)
        owner: Final = scenario.key(models=[model])
        response: Final = gateway.request("POST", "/v1/containers", {"model": model, "name": "dec1-bad"}, key=owner)
        assert response.status_code == 400, response.text
        assert "dec1 bad expiry" in response.text
        listed: Final = gateway.request("GET", "/v1/containers", key=owner, params={"model": model})
        assert listed.status_code == 200, listed.text
        assert list_value(JSON_OBJECT.validate_json(listed.content)["data"]) == [], listed.text
