import asyncio
from collections.abc import Callable, Coroutine
from datetime import datetime, timezone

import pytest
from fastapi import HTTPException

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.logging_endpoints.traces import get_span, get_trace, list_traces


def _list(auth: UserAPIKeyAuth) -> Coroutine[object, object, object]:
    return list_traces(
        start=datetime(2026, 1, 1, tzinfo=timezone.utc),
        end=datetime(2026, 1, 2, tzinfo=timezone.utc),
        limit=50,
        user_api_key_dict=auth,
    )


def _trace(auth: UserAPIKeyAuth) -> Coroutine[object, object, object]:
    return get_trace(trace_id="trace", user_api_key_dict=auth)


def _span(auth: UserAPIKeyAuth) -> Coroutine[object, object, object]:
    return get_span(trace_id="trace", span_id="span", user_api_key_dict=auth)


@pytest.mark.parametrize(
    "run_route",
    (
        _list,
        _trace,
        _span,
    ),
)
def test_trace_routes_reject_non_admin_callers_before_accessing_storage(
    run_route: Callable[[UserAPIKeyAuth], Coroutine[object, object, object]],
) -> None:
    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER)

    with pytest.raises(HTTPException) as error:
        asyncio.run(run_route(auth))

    assert error.value.status_code == 403
