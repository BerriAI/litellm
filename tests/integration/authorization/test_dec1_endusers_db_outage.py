"""What an end user's request gets when the database fails on that end user's row.

Contract, read from litellm/proxy/auth/auth_checks.py:2262-2350 `get_end_user_object` and
litellm/proxy/auth/user_api_key_auth.py:2862-2900 (the end user block of the auth builder, wrapped in
`except Exception`):

* An unknown end user must never make a request fail: when its lookup fails the request is served as if the end
  user had no row (fail open is right for an end user that restricts nothing).
* A customer whose row restricts something (a budget it is over, a model list) must not be served unrestricted
  just because its row could not be read. Today `get_end_user_object` turns every exception into "no row", so
  these are skipped as BUG, like the blocked customer case in `database/test_dec1_dbstores_lookup_outages.py`.

The rig is `lookup_outage` from `dec1_dbstores_support.py`: a private proxy on a scratch database behind a relay that
drops any connection whose outbound bytes contain a per-rig marker. Each end user id embeds the marker, so only
queries naming it fail. The relay is disarmed around setup and teardown.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import Final

import httpx
import pytest
from pydantic import JsonValue

from tests.integration._support.client import gateway_from_environment
from tests.integration.authorization.dec1_dbstores_support import (
    LookupOutage,
    assert_ok,
    chat_as,
    customer,
    forget_end_user,
    lookup_outage,
)

pytestmark: Final = pytest.mark.timeout(300)


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[LookupOutage]:
    with (
        gateway_from_environment() as shared,
        lookup_outage(
            shared,
            tmp_path_factory.mktemp("dec1-endusers-outage"),
            general_settings={"allow_requests_on_db_unavailable": False},
        ) as outage,
    ):
        yield outage


def _failing_id(rig: LookupOutage) -> str:
    return f"{rig.trigger}-dec1-endusers-{uuid.uuid4().hex[:8]}"


def _armed_chat(rig: LookupOutage, key: str, model: str, **extra: JsonValue) -> httpx.Response:
    rig.arm()
    try:
        return chat_as(rig.proxy, key, model, **extra)
    finally:
        rig.disarm()


def test_an_unknown_end_user_whose_lookup_fails_is_served(rig: LookupOutage) -> None:
    """A never seen end user id carrying the marker is sent as body `user` while every query naming it is dropped.

    The request is served with 200: a failed end user read is never an error for the caller. UNVERIFIED: whether the
    row lookup is attempted at all, since the restricted end user registry (`_end_user_is_known_unrestricted`,
    auth_checks.py:2310) may answer "unrestricted" first without a query naming the id.
    """
    model: Final = rig.scenario.model()
    key: Final = rig.scenario.key()
    end_user: Final = _failing_id(rig)
    forget_end_user(rig.scenario, end_user)
    assert_ok(_armed_chat(rig, key, model, user=end_user))


def test_a_customer_lookup_failure_does_not_serve_an_over_budget_customer(rig: LookupOutage) -> None:
    """A customer whose id carries the marker has `spend: 0.5` and `max_budget: 0.1`; its first request is sent while
    its row lookup is dropped.

    Intended: the request is refused (the 422 end user budget refusal, or a 503 database unavailable), never served.
    """
    pytest.skip(
        "BUG: get_end_user_object catches every exception and returns None, so a failed read of an over-budget "
        "customer's row serves the customer without its budget"
    )
    model: Final = rig.scenario.model()
    end_user: Final = customer(rig.scenario, user_id=_failing_id(rig), spend=0.5, max_budget=0.1)
    key: Final = rig.scenario.key()
    assert _armed_chat(rig, key, model, user=end_user).status_code in (422, 503)


def test_a_customer_lookup_failure_does_not_widen_a_customer_model_list(rig: LookupOutage) -> None:
    """A customer whose id carries the marker may only use model A; its first request asks for model B while its row
    lookup is dropped.

    Intended: the request is refused (403 `customer_model_access_denied`, or a 503), never served with model B.
    """
    pytest.skip(
        "BUG: get_end_user_object catches every exception and returns None, so a failed read of a model "
        "restricted customer's row serves any model the key allows"
    )
    allowed: Final = rig.scenario.model()
    other: Final = rig.scenario.model()
    end_user: Final = customer(rig.scenario, user_id=_failing_id(rig), models=[allowed])
    key: Final = rig.scenario.key()
    assert _armed_chat(rig, key, other, user=end_user).status_code in (403, 503)
