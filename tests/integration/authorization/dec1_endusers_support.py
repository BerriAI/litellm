"""Shared helpers for the `test_dec1_endusers_*` end user (customer) contract files.

End users are the people a caller acts for: named by the `x-litellm-customer-id` / `x-litellm-end-user-id`
headers, a configured header, the body `user` field, `litellm_metadata.user`, `metadata.user_id` or
`safety_identifier` (litellm/proxy/auth/auth_utils.py `get_end_user_id_from_request_body`). Their rows live in
`LiteLLM_EndUserTable`; their budget and limits live in the linked `LiteLLM_BudgetTable` row. The budget engine
(#51535), the v3 limiter (#51537) and the end user store (draft #52170) are being ported to Rust, so these helpers
only drive the proxy over HTTP and read Postgres, Redis or the scripted upstream to observe an effect.

Everything that already exists in `dec1_dbstores_support.py`, `dec1_keys_support.py` and `dec1_limits_support.py`
is imported there, not copied here.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Final

import httpx
from pydantic import JsonValue

from tests.integration._support.client import Gateway, eventually, gateway_from_environment, object_value
from tests.integration._support.process import owned_proxy
from tests.integration.authorization.dec1_dbstores_support import config_with_settings
from tests.integration.authorization.dec1_keys_support import digest, error_of, upstream_requests

#: `litellm.budget_exceeded_status_code` (litellm/__init__.py), the status of every budget refusal.
BUDGET_EXCEEDED_STATUS: Final = 422
#: litellm/constants.py:1723 `STANDARD_CUSTOMER_ID_HEADERS`, checked in this order.
CUSTOMER_HEADER: Final = "x-litellm-customer-id"
END_USER_HEADER: Final = "x-litellm-end-user-id"
#: litellm/constants.py:1820, what `overwrite_user_with_key_hash` stamps for the master key.
MASTER_KEY_ALIAS: Final = "litellm_proxy_master_key"
#: The spend row and its end user row land asynchronously; the suite allows 70 seconds for spend rows.
SPEND_SECONDS: Final = 70


def new_end_user() -> str:
    """A never seen end user id; every id carries the file prefix so a leaked row is easy to trace."""
    return f"dec1-endusers-{uuid.uuid4().hex}"


def text_marker() -> str:
    """A unique prompt, so the scripted upstream's observations can be filtered to one request."""
    return f"dec1-endusers-prompt-{uuid.uuid4().hex}"


def spend_rows(gateway: Gateway, key: str, *, at_least: int = 1) -> list[dict[str, JsonValue]]:
    """Every `LiteLLM_SpendLogs` row of `key`, oldest first, once `at_least` have landed."""
    return eventually(
        lambda: gateway.rows(
            'SELECT request_id, end_user, "user", spend, status, metadata, call_type FROM "LiteLLM_SpendLogs" '
            'WHERE api_key=%s ORDER BY "startTime"',
            (digest(key),),
        ),
        lambda rows: len(rows) >= at_least,
        seconds=SPEND_SECONDS,
    )


def end_user_row(gateway: Gateway, end_user: str) -> dict[str, JsonValue] | None:
    """The whole `LiteLLM_EndUserTable` row of `end_user`, or None when there is none."""
    rows: Final = gateway.rows(
        "SELECT user_id, alias, spend, blocked, budget_id, allowed_model_region, default_model, "
        'object_permission_id FROM "LiteLLM_EndUserTable" WHERE user_id=%s',
        (end_user,),
    )
    return rows[0] if rows else None


def landed_end_user_row(gateway: Gateway, end_user: str, *, min_spend: float = 0.0) -> dict[str, JsonValue]:
    """Poll until the end user row exists with `spend` above `min_spend` (the spend writer is batched)."""
    row: Final = eventually(
        lambda: end_user_row(gateway, end_user),
        lambda found: found is not None and float(str(found["spend"])) > min_spend,
        seconds=SPEND_SECONDS,
    )
    assert row is not None
    return row


def upstream_bodies(gateway: Gateway, text: str) -> list[dict[str, JsonValue]]:
    """The JSON bodies the scripted upstream received for the request whose prompt is `text`."""
    return [object_value(request["body"]) for request in upstream_requests(gateway, text)]


def assert_end_user_over_budget(response: httpx.Response, end_user: str, spend: str, budget: str) -> None:
    """The auth refusal of an end user above its budget (litellm/rust/crates/callbacks-budgets/src/checks.rs:430-452
    renders the text, litellm/proxy/auth/auth_checks.py:2010 `check_end_user_budget` raises it): 422, type
    `budget_exceeded`, code "422", param None."""
    assert response.status_code == BUDGET_EXCEEDED_STATUS, f"{response.status_code} {response.text}"
    error: Final = error_of(response)
    assert error["message"] == f"ExceededBudget: End User={end_user} over budget. Spend={spend}, Budget={budget}", (
        response.text
    )
    assert (error["type"], error["param"], error["code"]) == ("budget_exceeded", None, "422"), response.text


def owned_endusers_proxy(
    directory: Path,
    *,
    general_settings: Mapping[str, JsonValue] | None = None,
    litellm_settings: Mapping[str, JsonValue] | None = None,
) -> Iterator[Gateway]:
    """A private proxy on the suite database and upstream with extra settings merged over the suite config."""
    config: Final = config_with_settings(
        directory, general_settings=general_settings, litellm_settings=litellm_settings, name="dec1-endusers.yaml"
    )
    with gateway_from_environment() as shared, owned_proxy(shared, directory / "proxy", {}, config=config) as proxy:
        yield proxy
