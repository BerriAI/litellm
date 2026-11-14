"""What an authenticated key adds to, or demands from, the request after auth has passed.

Contract (litellm/proxy/litellm_pre_call_utils.py `add_user_api_key_auth_to_request_metadata`,
`add_key_level_controls`, `_enforced_params_check`; litellm/proxy/hooks/cache_control_check.py):

* The resolved identity (key hash, key alias, user, team, organization, end user) is stamped onto the request
  and lands on the `LiteLLM_SpendLogs` row. Values a client puts in the body under the same names are
  overwritten, so a caller cannot attribute spend to another team or key.
* Tags in the key's and the team's `metadata.tags` are merged into the request's tags.
* `spend_logs_metadata` on the key is merged under what the request sent; the request wins on a clash.
* `enforced_params` on the key makes the listed body params mandatory.
* `allowed_cache_controls` on the key limits which `cache` controls a request may set (403 otherwise).
* The key's limits are echoed on responses as `x-litellm-key-tpm-limit`, `x-litellm-key-rpm-limit` and
  `x-litellm-key-max-budget`.
* None of this identity data is forwarded to the upstream provider.

Spend rows are written asynchronously, so every row read polls.
"""

from __future__ import annotations

from typing import Final

from tests.integration._support.client import Gateway, list_value, object_value
from tests.integration.authorization.dec1_keys_support import (
    assert_denied,
    assert_served,
    chat,
    digest,
    key_row,
    marker,
    spend_rows,
    upstream_requests,
)


def test_the_spend_row_carries_the_key_hash_alias_user_and_team(gateway: Gateway) -> None:
    """A key with an alias, owned by a team member inside a team, serves one chat completion.

    Its spend row must have `api_key` equal to the SHA-256 of the key, `user` equal to the key's user id,
    `team_id` equal to the team, and metadata `user_api_key_alias`, `user_api_key_team_id` and
    `user_api_key_user_id` matching. The raw key must not appear in the row.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team()
        member: Final = scenario.member(team)
        alias: Final = marker()
        key: Final = scenario.key(team_id=team, user_id=member, key_alias=alias)
        assert_served(chat(gateway, key, model))
        row: Final = spend_rows(gateway, key)[0]
        metadata: Final = object_value(row["metadata"])
        assert (row["api_key"], row["user"], row["team_id"]) == (digest(key), member, team), row
        assert (
            metadata["user_api_key_alias"],
            metadata["user_api_key_team_id"],
            metadata["user_api_key_user_id"],
        ) == (alias, team, member), metadata
        assert key not in str(row), row


def test_the_spend_row_of_an_organization_team_key_carries_the_organization(gateway: Gateway) -> None:
    """A key belongs to a team that belongs to an organization and serves one chat completion.

    The key row itself has no organization, so auth must inherit it from the team: the spend row's
    `organization_id` must equal the team's organization.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        organization: Final = scenario.organization()
        team: Final = scenario.team(organization_id=organization)
        key: Final = scenario.key(team_id=team)
        assert_served(chat(gateway, key, model))
        row: Final = spend_rows(gateway, key)[0]
        assert (row["team_id"], row["organization_id"]) == (team, organization), row


def test_the_body_user_field_is_recorded_as_the_end_user(gateway: Gateway) -> None:
    """A key serves a chat completion whose body carries `user: <customer id>`.

    The spend row's `end_user` must equal that customer id, and the customer id must not be recorded as
    the row's `user`, which is reserved for the key's internal user.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        customer: Final = marker()
        assert_served(chat(gateway, key, model, user=customer))
        row: Final = spend_rows(gateway, key)[0]
        assert row["end_user"] == customer, row
        assert row["user"] != customer, row


def test_a_client_cannot_overwrite_the_resolved_identity_through_body_metadata(gateway: Gateway) -> None:
    """A team key sends a chat completion whose body `metadata` claims another team, user, alias and key hash.

    The spend row must still carry the real team and key hash, and its metadata must carry the real
    `user_api_key_team_id` and `user_api_key_alias`. The spoofed values must not appear in those fields.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team()
        alias: Final = marker()
        key: Final = scenario.key(team_id=team, key_alias=alias)
        spoof: Final = f"spoof-{marker()}"
        assert_served(
            chat(
                gateway,
                key,
                model,
                metadata={
                    "user_api_key_team_id": spoof,
                    "user_api_key_alias": spoof,
                    "user_api_key_user_id": spoof,
                    "user_api_key": spoof,
                    "user_api_key_hash": spoof,
                },
            )
        )
        row: Final = spend_rows(gateway, key)[0]
        metadata: Final = object_value(row["metadata"])
        assert (row["api_key"], row["team_id"]) == (digest(key), team), row
        assert (metadata["user_api_key_team_id"], metadata["user_api_key_alias"]) == (team, alias), metadata
        assert spoof not in (row["user"], row["team_id"], row["api_key"]), row


def test_identity_metadata_is_not_forwarded_to_the_upstream(gateway: Gateway) -> None:
    """A team key with an alias serves a chat completion and the forwarded upstream request is inspected.

    The upstream body must not contain the key hash, the key alias, the team id or any `user_api_key`
    field. Identity is for logging and policy inside the proxy only.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team()
        alias: Final = marker()
        key: Final = scenario.key(team_id=team, key_alias=alias)
        text: Final = marker()
        assert_served(chat(gateway, key, model, text))
        forwarded: Final = upstream_requests(gateway, text)
        assert len(forwarded) == 1, forwarded
        rendered: Final = str(forwarded[0])
        assert [leak for leak in (digest(key), alias, team, "user_api_key") if leak in rendered] == [], rendered


def test_tags_in_key_metadata_are_added_to_the_request(gateway: Gateway) -> None:
    """A key is created with `metadata: {"tags": [<tag>]}` and serves a chat completion without tags.

    The spend row's `request_tags` must contain the key's tag.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        tag: Final = f"key-tag-{marker()}"
        key: Final = scenario.key(metadata={"tags": [tag]})
        assert_served(chat(gateway, key, model))
        assert tag in list_value(spend_rows(gateway, key)[0]["request_tags"])


def test_key_tags_are_merged_with_tags_the_client_sends(gateway: Gateway) -> None:
    """A key with a metadata tag serves a chat completion whose body `metadata.tags` carries a client tag.

    The spend row's `request_tags` must contain both tags, each once. The key's tag must not replace the
    client's, and the client's must not drop the key's.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key_tag: Final = f"key-tag-{marker()}"
        client_tag: Final = f"client-tag-{marker()}"
        key: Final = scenario.key(metadata={"tags": [key_tag]})
        assert_served(chat(gateway, key, model, metadata={"tags": [client_tag, key_tag]}))
        tags: Final = list_value(spend_rows(gateway, key)[0]["request_tags"])
        assert (tags.count(key_tag), tags.count(client_tag)) == (1, 1), tags


def test_team_metadata_tags_are_added_alongside_key_tags(gateway: Gateway) -> None:
    """A team has `metadata.tags` and its key has its own `metadata.tags`; the key serves a chat completion.

    The spend row's `request_tags` must contain both the key's tag and the team's tag.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team_tag: Final = f"team-tag-{marker()}"
        key_tag: Final = f"key-tag-{marker()}"
        team: Final = scenario.team(metadata={"tags": [team_tag]})
        key: Final = scenario.key(team_id=team, metadata={"tags": [key_tag]})
        assert_served(chat(gateway, key, model))
        tags: Final = list_value(spend_rows(gateway, key)[0]["request_tags"])
        assert key_tag in tags and team_tag in tags, tags


def test_key_spend_logs_metadata_is_merged_under_the_requests_values(gateway: Gateway) -> None:
    """A key has `metadata.spend_logs_metadata: {"cost_center": "key", "owner": "key"}` and the request sends
    `metadata.spend_logs_metadata: {"cost_center": "request"}`.

    The spend row's `metadata.spend_logs_metadata` must be `{"cost_center": "request", "owner": "key"}`:
    the key fills in what the request did not send and never overrides what it did.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(metadata={"spend_logs_metadata": {"cost_center": "key", "owner": "key"}})
        assert_served(chat(gateway, key, model, metadata={"spend_logs_metadata": {"cost_center": "request"}}))
        metadata: Final = object_value(spend_rows(gateway, key)[0]["metadata"])
        assert metadata["spend_logs_metadata"] == {"cost_center": "request", "owner": "key"}, metadata


def test_key_spend_logs_metadata_is_used_when_the_request_sends_none(gateway: Gateway) -> None:
    """A key has `metadata.spend_logs_metadata: {"owner": "key"}` and the request sends no metadata.

    The spend row's `metadata.spend_logs_metadata` must equal `{"owner": "key"}`.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(metadata={"spend_logs_metadata": {"owner": "key"}})
        assert_served(chat(gateway, key, model))
        metadata: Final = object_value(spend_rows(gateway, key)[0]["metadata"])
        assert metadata["spend_logs_metadata"] == {"owner": "key"}, metadata


def test_an_enforced_param_missing_from_the_body_is_refused_before_the_upstream(gateway: Gateway) -> None:
    """A key is created with `enforced_params: ["user"]` and sends a chat completion without `user`.

    The request must fail with a message containing "BadRequest please pass param=user in request body.
    This is a required param" and the upstream must not be called. UNVERIFIED: the status code. The check
    raises a plain `ValueError` in `add_litellm_data_to_request`, expected to surface as a 4xx or 500.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(enforced_params=["user"])
        text: Final = marker()
        refused: Final = chat(gateway, key, model, text)
        assert refused.status_code >= 400, refused.text
        assert "BadRequest please pass param=user in request body. This is a required param" in refused.text
        assert upstream_requests(gateway, text) == []


def test_an_enforced_param_present_in_the_body_is_served(gateway: Gateway) -> None:
    """A key with `enforced_params: ["user"]` sends a chat completion that includes `user`.

    The request must be served with 200.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(enforced_params=["user"])
        assert_served(chat(gateway, key, model, user=marker()))


def test_a_nested_enforced_param_names_the_missing_field_in_brackets(gateway: Gateway) -> None:
    """A key with `enforced_params: ["metadata.generation_name"]` sends `metadata: {}` and then a complete body.

    The first request must fail with a message containing "BadRequest please pass
    param=[metadata][generation_name] in request body. This is a required param". The second, with
    `metadata.generation_name` set, must be served with 200.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(enforced_params=["metadata.generation_name"])
        refused: Final = chat(gateway, key, model, metadata={"other": "value"})
        assert refused.status_code >= 400, refused.text
        assert (
            "BadRequest please pass param=[metadata][generation_name] in request body. This is a required param"
            in refused.text
        ), refused.text
        assert_served(chat(gateway, key, model, metadata={"generation_name": "dec1"}))


def test_enforced_params_on_one_key_do_not_apply_to_another_key(gateway: Gateway) -> None:
    """Two keys exist, one with `enforced_params: ["user"]` and one without; both omit `user`.

    The plain key must be served with 200 while the enforcing key is refused. The requirement is per key,
    not proxy wide.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        strict: Final = scenario.key(enforced_params=["user"])
        plain: Final = scenario.key()
        assert chat(gateway, strict, model).status_code >= 400
        assert_served(chat(gateway, plain, model))


def test_a_cache_control_outside_the_keys_allow_list_is_a_403(gateway: Gateway) -> None:
    """A key is created with `allowed_cache_controls: ["no-cache"]` and sends `cache: {"no-store": true}`.

    The proxy must answer 403 with code "403", type `permission_error` and the message "Not allowed to set
    no-store as a cache control. Contact admin to change permissions.", and the upstream must not be called.
    The pre call hook raises a bare `HTTPException(403)`, and the route maps a 403 without its own type to
    `permission_error` (litellm/proxy/common_utils/openai_error_payload.py `_OPENAI_ERROR_TYPE_BY_STATUS`).
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(allowed_cache_controls=["no-cache"])
        text: Final = marker()
        refused: Final = chat(gateway, key, model, text, cache={"no-store": True})
        assert_denied(
            refused,
            status=403,
            error_type="permission_error",
            message="Not allowed to set no-store as a cache control. Contact admin to change permissions.",
        )
        assert upstream_requests(gateway, text) == []


def test_a_cache_control_inside_the_keys_allow_list_is_served(gateway: Gateway) -> None:
    """A key with `allowed_cache_controls: ["no-cache"]` sends `cache: {"no-cache": true}`.

    The control is on the key's list, so the request must be served with 200.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(allowed_cache_controls=["no-cache"])
        assert_served(chat(gateway, key, model, cache={"no-cache": True}))


def test_a_key_without_allowed_cache_controls_may_set_any_cache_control(gateway: Gateway) -> None:
    """A key created without `allowed_cache_controls` sends `cache: {"no-store": true}`.

    An empty list means unrestricted, so the request must be served with 200.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        assert_served(chat(gateway, key, model, cache={"no-store": True}))


def test_one_disallowed_cache_control_refuses_the_whole_request(gateway: Gateway) -> None:
    """A key allowed only `no-cache` sends `cache: {"no-cache": true, "ttl": 60}`.

    One control is allowed and one is not, and the request must be refused with 403 naming `ttl`.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(allowed_cache_controls=["no-cache"])
        refused: Final = chat(gateway, key, model, cache={"no-cache": True, "ttl": 60})
        assert refused.status_code == 403, refused.text
        assert "Not allowed to set ttl as a cache control." in refused.text, refused.text


def test_the_keys_limits_are_echoed_in_response_headers(gateway: Gateway) -> None:
    """A key with `tpm_limit: 12345`, `rpm_limit: 67` and `max_budget: 8.5` serves a chat completion.

    The response must carry `x-litellm-key-tpm-limit: 12345`, `x-litellm-key-rpm-limit: 67` and
    `x-litellm-key-max-budget: 8.5`, and an `x-litellm-key-spend` header that parses as a number.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(tpm_limit=12345, rpm_limit=67, max_budget=8.5)
        response: Final = chat(gateway, key, model)
        assert_served(response)
        assert (
            response.headers["x-litellm-key-tpm-limit"],
            response.headers["x-litellm-key-rpm-limit"],
            response.headers["x-litellm-key-max-budget"],
        ) == ("12345", "67", "8.5"), dict(response.headers)
        assert float(response.headers["x-litellm-key-spend"]) >= 0.0


def test_a_key_without_limits_omits_each_limit_header(gateway: Gateway) -> None:
    """A key created without limits serves a chat completion.

    The three limit headers must be absent. `get_custom_headers` renders each limit with `str()` and then drops
    every header whose value is `""`, `None` or `"None"` (litellm/proxy/common_request_processing.py
    `exclude_values`), so an unset limit is never sent. The spend header is still present.
    """
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key()
        response: Final = chat(gateway, key, model)
        assert_served(response)
        absent: Final = ("x-litellm-key-tpm-limit", "x-litellm-key-rpm-limit", "x-litellm-key-max-budget")
        assert [name for name in absent if name in response.headers] == [], dict(response.headers)
        assert float(response.headers["x-litellm-key-spend"]) >= 0.0


def test_guardrails_given_at_key_creation_are_stored_in_key_metadata(gateway: Gateway) -> None:
    """A key is generated with `guardrails: [<name>]` and its row is read.

    The row's `metadata.guardrails` must equal that list. Auth reads guardrails from key metadata on every
    request, so this stored location is the contract between key management and the request path.
    """
    with gateway.scenario() as scenario:
        name: Final = f"guardrail-{marker()}"
        key: Final = scenario.key(guardrails=[name])
        assert object_value(key_row(gateway, key)["metadata"])["guardrails"] == [name]


def test_enforced_params_given_at_key_creation_are_stored_in_key_metadata(gateway: Gateway) -> None:
    """A key is generated with `enforced_params: ["user"]` and its row is read.

    The row's `metadata.enforced_params` must equal `["user"]`, which is where the request path reads it.
    """
    with gateway.scenario() as scenario:
        key: Final = scenario.key(enforced_params=["user"])
        assert object_value(key_row(gateway, key)["metadata"])["enforced_params"] == ["user"]
