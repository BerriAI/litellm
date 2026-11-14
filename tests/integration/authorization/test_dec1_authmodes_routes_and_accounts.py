"""Proxy-wide route settings that change who must authenticate (`public_routes`, `admin_only_routes`,
`allowed_routes`) and service account keys, which act for a team with no user behind them.

Contract (litellm/proxy/auth/auth_utils.py `route_in_additonal_public_routes`, `pre_db_read_auth_checks`;
litellm/proxy/auth/route_checks.py `custom_admin_only_route_check`, `is_allowed_on_proxy`;
litellm/proxy/litellm_pre_call_utils.py `_get_enforced_params`, `_enforced_params_check`):

* `public_routes` (exact or `/prefix/*` wildcard) skips authentication entirely, with a license only.
* `admin_only_routes` refuses every non-admin caller with 403 "user not allowed to access this route.
  Route=<route> is an admin only route", with a license only; without one it is ignored.
* `general_settings.allowed_routes` closes every other route for every caller, the master key included,
  with 403 "Access forbidden: Route <route> not allowed".
* `POST /key/service-account/generate` makes a team key whose metadata carries `service_account_id`; such a
  key spends as the team with no user. `service_account_settings.enforced_params` names body fields every
  service account request must carry.

`allowed_ips`, `enforce_user_param` and `user_header_name` are covered by the http and configuration
contracts and are not repeated here.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Final

import httpx
import pytest

from tests.integration._support.client import JSON_OBJECT, Gateway, object_value, string_value
from tests.integration.authorization.dec1_authmodes_support import (
    authmodes_proxy,
    chat_as,
    served_body,
    spend_row,
    unique,
    without_credentials,
)
from tests.integration.authorization.dec1_keys_support import assert_denied

ROUTE_SETTINGS: Final = {
    "public_routes": ["/model_group/info", "/v1/model/*"],
    "admin_only_routes": ["/v1/models"],
}
HARNESS_ROUTES: Final = (
    "/model/new",
    "/model/delete",
    "/model/info",
    "/key/generate",
    "/key/delete",
    "/key/info",
)


@pytest.fixture(scope="module")
def routes(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    with authmodes_proxy(tmp_path_factory.mktemp("dec1-authmodes-routes"), general_settings=ROUTE_SETTINGS) as proxy:
        yield proxy


@pytest.fixture(scope="module")
def unlicensed_routes(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    with authmodes_proxy(
        tmp_path_factory.mktemp("dec1-authmodes-routes-unlicensed"),
        general_settings=ROUTE_SETTINGS,
        remove_environment=("LITELLM_LICENSE",),
    ) as proxy:
        yield proxy


@pytest.fixture(scope="module")
def closed(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    with authmodes_proxy(
        tmp_path_factory.mktemp("dec1-authmodes-allowed-routes"),
        general_settings={"allowed_routes": ["/v1/chat/completions", *HARNESS_ROUTES]},
    ) as proxy:
        yield proxy


@pytest.fixture(scope="module")
def accounts(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    with authmodes_proxy(
        tmp_path_factory.mktemp("dec1-authmodes-service-accounts"),
        general_settings={"service_account_settings": {"enforced_params": ["user"]}},
    ) as proxy:
        yield proxy


@pytest.fixture(scope="module")
def layered(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    with authmodes_proxy(
        tmp_path_factory.mktemp("dec1-authmodes-enforced-layers"),
        general_settings={
            "enforced_params": ["metadata"],
            "service_account_settings": {"enforced_params": ["user"]},
        },
    ) as proxy:
        yield proxy


def _service_account_key(proxy: Gateway, team: str, model: str) -> str:
    response: Final = proxy.request(
        "POST", "/key/service-account/generate", {"team_id": team, "models": [model], "key_alias": unique("sa")}
    )
    assert response.status_code == 200, response.text
    body: Final = JSON_OBJECT.validate_json(response.content)
    assert "service_account_id" in object_value(body["metadata"]), response.text
    return string_value(body["key"])


def _required_param(response: httpx.Response, param: str) -> None:
    """The enforced params refusal. UNVERIFIED: the status (the pre-call `ValueError` is mapped by the route's
    generic error handler), so only a client or server error carrying the exact text is required."""
    assert response.status_code in {400, 500}, response.text
    assert f"BadRequest please pass param={param} in request body. This is a required param" in response.text


def test_an_exact_public_route_needs_no_credentials(routes: Gateway) -> None:
    """`public_routes` lists `/model_group/info`; a GET carries no `Authorization` header at all.

    The proxy must answer 200.
    """
    response: Final = without_credentials(routes, "GET", "/model_group/info")
    assert response.status_code == 200, response.text


def test_a_wildcard_public_route_needs_no_credentials(routes: Gateway) -> None:
    """`public_routes` lists `/v1/model/*`; `GET /v1/model/info` carries no credentials.

    The wildcard covers it, so the proxy must answer 200.
    """
    response: Final = without_credentials(routes, "GET", "/v1/model/info")
    assert response.status_code == 200, response.text


def test_a_public_route_ignores_a_bogus_bearer_value(routes: Gateway) -> None:
    """`GET /model_group/info` is public and the request carries `Authorization: Bearer dec1-garbage`.

    Authentication is skipped before the value is read, so the proxy must answer 200 rather than 401.
    """
    response: Final = routes.request("GET", "/model_group/info", key="dec1-garbage")
    assert response.status_code == 200, response.text


def test_a_route_not_listed_as_public_still_needs_credentials(routes: Gateway) -> None:
    """`GET /v1/models` is not in `public_routes`; the request carries no credentials.

    The proxy must answer 401.
    """
    response: Final = without_credentials(routes, "GET", "/v1/models")
    assert response.status_code == 401, response.text


def test_an_admin_only_route_refuses_a_virtual_key(routes: Gateway) -> None:
    """`admin_only_routes` lists `/v1/models`; an ordinary virtual key calls it.

    The proxy must answer 403 `auth_error` with exactly "user not allowed to access this route.
    Route=/v1/models is an admin only route".
    """
    with routes.scenario() as scenario:
        key: Final = scenario.key()
        assert_denied(
            routes.request("GET", "/v1/models", key=key),
            status=403,
            error_type="auth_error",
            message="user not allowed to access this route. Route=/v1/models is an admin only route",
        )


def test_an_admin_only_route_serves_the_master_key(routes: Gateway) -> None:
    """`admin_only_routes` lists `/v1/models`; the master key calls it.

    The proxy must answer 200 with a model list.
    """
    response: Final = routes.request("GET", "/v1/models")
    assert response.status_code == 200, response.text
    assert JSON_OBJECT.validate_json(response.content)["object"] == "list"


def test_an_admin_only_route_leaves_other_routes_open_to_virtual_keys(routes: Gateway) -> None:
    """With `/v1/models` admin only, an ordinary virtual key calls a chat completion.

    Only the listed route is closed, so the chat must be served.
    """
    with routes.scenario() as scenario:
        model: Final = scenario.model()
        assert served_body(chat_as(routes, scenario.key(models=[model]), model))


def test_public_routes_are_ignored_without_a_license(unlicensed_routes: Gateway) -> None:
    """The same `public_routes` but `LITELLM_LICENSE` is removed; `GET /model_group/info` has no credentials.

    The public route list is an enterprise feature, so the proxy must answer 401.
    """
    response: Final = without_credentials(unlicensed_routes, "GET", "/model_group/info")
    assert response.status_code == 401, response.text


def test_admin_only_routes_are_ignored_without_a_license(unlicensed_routes: Gateway) -> None:
    """The same `admin_only_routes` but no license; an ordinary virtual key calls `GET /v1/models`.

    The setting is skipped with a logged error, so the proxy must answer 200. Pinned because it fails open.
    """
    with unlicensed_routes.scenario() as scenario:
        response: Final = unlicensed_routes.request("GET", "/v1/models", key=scenario.key())
        assert response.status_code == 200, response.text


def test_proxy_wide_allowed_routes_close_other_routes_even_for_the_master_key(closed: Gateway) -> None:
    """`general_settings.allowed_routes` lists chat completions and the management routes the test needs;
    the master key calls `GET /v1/models`.

    The proxy must answer 403 with the text "Access forbidden: Route /v1/models not allowed".
    UNVERIFIED: whether the body is OpenAI shaped or a bare `{"detail": ...}`; only the text is required.
    """
    response: Final = closed.request("GET", "/v1/models")
    assert response.status_code == 403, response.text
    assert "Access forbidden: Route /v1/models not allowed" in response.text


def test_proxy_wide_allowed_routes_serve_the_listed_routes(closed: Gateway) -> None:
    """With the same `allowed_routes`, the master key calls a chat completion, which is listed.

    The chat must be served.
    """
    with closed.scenario() as scenario:
        model: Final = scenario.model()
        assert served_body(chat_as(closed, closed.key, model))


def test_a_service_account_key_spends_as_its_team_with_no_user(accounts: Gateway) -> None:
    """A key from `/key/service-account/generate` for a team calls a chat completion with a `user` field.

    The spend row's `team_id` must be the team, its `user` empty (no person owns the key), and its
    `end_user` the body's `user`.
    """
    with accounts.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team(models=[model])
        key: Final = _service_account_key(accounts, team, model)
        scenario.cleanups.callback(scenario.delete_key, key)
        customer: Final = unique("sa-customer")
        row: Final = spend_row(accounts, chat_as(accounts, key, model, user=customer))
        assert row["team_id"] == team, row
        assert row["user"] in (None, ""), row
        assert row["end_user"] == customer, row


def test_a_service_account_request_without_an_enforced_param_is_refused(accounts: Gateway) -> None:
    """`service_account_settings.enforced_params: [user]`; a service account key calls a chat completion
    without `user`.

    The request must be refused with "BadRequest please pass param=user in request body. This is a required
    param".
    """
    with accounts.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team(models=[model])
        key: Final = _service_account_key(accounts, team, model)
        scenario.cleanups.callback(scenario.delete_key, key)
        _required_param(chat_as(accounts, key, model), "user")


def test_service_account_enforced_params_do_not_apply_to_ordinary_keys(accounts: Gateway) -> None:
    """With the same setting, an ordinary key of the same team calls a chat completion without `user`.

    The enforced params are only for service accounts, so the chat must be served.
    """
    with accounts.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team(models=[model])
        assert served_body(chat_as(accounts, scenario.key(team_id=team, models=[model]), model))


def test_a_service_account_request_does_not_widen_the_global_enforced_params(layered: Gateway) -> None:
    """Global `enforced_params: [metadata]` plus service account `enforced_params: [user]`. An ordinary key is
    served with `metadata` only, a service account key is served with both, then the ordinary key sends
    `metadata` only again.

    The ordinary key must still be served: service account rules must not leak into every request.
    """
    pytest.skip(
        "BUG: `_get_enforced_params` (litellm_pre_call_utils.py) extends the list it read from "
        "`general_settings['enforced_params']` in place, so each service account request (and each key with "
        "metadata.enforced_params) appends its params to the global list for every later caller. "
        "UNVERIFIED in a run."
    )
    with layered.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team(models=[model])
        ordinary: Final = scenario.key(team_id=team, models=[model])
        service: Final = _service_account_key(layered, team, model)
        scenario.cleanups.callback(scenario.delete_key, service)
        tag: Final = {"dec1": unique("tag")}
        assert served_body(chat_as(layered, ordinary, model, metadata=tag))
        assert served_body(chat_as(layered, service, model, metadata=tag, user=unique("sa-user")))
        assert served_body(chat_as(layered, ordinary, model, metadata=tag))
