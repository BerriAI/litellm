"""Batch creation under rate limits: the batch limiter and the enqueued token allowance.

``POST /v1/batches`` does not go through the per-request v3 check. ``get_rate_limiter_for_call_type`` hands
``acreate_batch`` to ``_PROXY_BatchRateLimiter`` (parallel_request_limiter_v3.py:3483), which reads the input file,
counts its rows and tokens, and charges the whole batch at once (``litellm/proxy/hooks/batch_rate_limiter.py``):

- per-minute key, team and user limits are charged the batch's request count and token count, refused with
  ``Batch rate limit exceeded for <scope>: <value>. Batch contains <n> requests but only <r> requests remaining out of
  <limit> RPM limit. Limit resets at: <UTC>``;
- a scope with ``tpd_limit`` (key, team, end user) is charged on a separate daily descriptor ``<scope>_tpd`` instead
  of its per-minute one, with ``TPD limit`` in the text;
- ``metadata.batch_enqueued_token_limit`` on a key or team (admin set) replaces both with a long-lived allowance that
  is refunded when the batch reaches a terminal state, refused with ``Batch enqueued token limit exceeded for
  <scope>: <value>. ...`` (``_raise_enqueued_limit_error``, batch_rate_limiter.py:734);
- ``model_batch_enqueued_token_limit`` in project metadata does the same per model group (#51608), pinned for custom
  auth callers in ``spend/test_project_batch_enqueued_token_limit.py``.

Each test owns a stateful batch provider (``BatchProvider``) registered as a model on the shared gateway, so the
provider's ``POST /batches`` count shows whether the proxy forwarded the create. Token counts depend on the token
counter, so token numbers are matched as integers and only the limit and scope are literal.
"""

from __future__ import annotations

import re
from typing import Final

from tests.integration._support.client import Gateway, eventually
from tests.integration._support.wire import Wire, wire_server
from tests.integration.authorization.dec1_limcap2_support import BatchProvider, batch_id, submit_batch
from tests.integration.authorization.dec1_limits_support import error_message, hashed, redis_integer, shared_redis

RESET_AT: Final = r"Limit resets at: \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC"


def _creates(wire: Wire) -> int:
    return sum(
        1
        for request in wire.drain()
        if request.method == "POST" and request.target.split("?", 1)[0].endswith("/batches")
    )


def test_a_batch_with_more_rows_than_the_key_rpm_is_refused_whole(gateway: Gateway) -> None:
    """A key with ``rpm_limit: 10`` creates a batch of 12 rows. The batch limiter charges all 12 requests at once, so
    the create is refused 429 with ``Batch rate limit exceeded for api_key: <hash>. Batch contains 12 requests but
    only <r> requests remaining out of 10 RPM limit. Limit resets at: <UTC>`` and the provider never receives
    ``POST /batches``. The upload itself may use one of the ten, so ``<r>`` is matched, not pinned."""
    with wire_server(BatchProvider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], rpm_limit=10)
        refused: Final = submit_batch(gateway, model, key, rows=12)
        assert refused.status_code == 429, refused.text
        assert re.fullmatch(
            rf"Batch rate limit exceeded for api_key: {hashed(key)}\. Batch contains 12 requests but only \d+ "
            rf"requests remaining out of 10 RPM limit\. {RESET_AT}",
            error_message(refused),
        ), refused.text
        assert refused.headers["rate_limit_type"] == "requests", dict(refused.headers)
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC", refused.headers["reset_at"]), dict(
            refused.headers
        )
        assert _creates(wire) == 0


def test_a_batch_that_fits_the_key_rpm_is_forwarded_and_charged_its_row_count(gateway: Gateway) -> None:
    """A key with ``rpm_limit: 10`` creates a batch of 3 rows and is served 200 (the provider receives one
    ``POST /batches``). The key's request counter then holds at least 3 more than before the create, because every row
    is charged (``_check_and_increment_batch_counters``), so a following 8-row batch is refused in the same minute."""
    with wire_server(BatchProvider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], rpm_limit=10)
        created: Final = submit_batch(gateway, model, key, rows=3)
        assert created.status_code == 200, created.text
        assert _creates(wire) == 1
        refused: Final = submit_batch(gateway, model, key, rows=8)
        assert refused.status_code == 429, refused.text
        assert "Batch contains 8 requests" in error_message(refused), refused.text


def test_a_key_tpd_limit_charges_the_batch_on_a_daily_descriptor_with_a_tpd_text(gateway: Gateway) -> None:
    """A key with ``tpd_limit: 50`` creates a batch of 3 rows of about 40 tokens each. The batch is charged on the
    daily descriptor ``api_key_tpd`` (``BATCH_TPD_DESCRIPTOR_SUFFIX``, window 86400 s) and refused with ``Batch rate
    limit exceeded for api_key_tpd: <hash>. Batch contains <n> tokens but only <r> tokens remaining out of 50 TPD
    limit.``. Its ``retry-after`` is at most a day, not the minute window."""
    with wire_server(BatchProvider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], tpd_limit=50)
        refused: Final = submit_batch(gateway, model, key, rows=3, words=40)
        assert refused.status_code == 429, refused.text
        assert re.fullmatch(
            rf"Batch rate limit exceeded for api_key_tpd: {hashed(key)}\. Batch contains \d+ tokens but only \d+ "
            rf"tokens remaining out of 50 TPD limit\. {RESET_AT}",
            error_message(refused),
        ), refused.text
        assert 0 <= int(refused.headers["retry-after"]) <= 86400, dict(refused.headers)
        assert _creates(wire) == 0


def test_a_key_tpd_limit_takes_the_batch_off_the_per_minute_token_limit(gateway: Gateway) -> None:
    """A key has ``tpm_limit: 5`` and ``tpd_limit: 100000``. A 3 row batch of far more than 5 tokens is accepted 200:
    with a ``tpd_limit`` the key's per-minute descriptor is dropped from the batch charge and replaced by the daily one
    (``_create_batch_rate_limit_descriptors``, batch_rate_limiter.py:341), because batch rows never run in the
    submission's minute."""
    with wire_server(BatchProvider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], tpm_limit=5, tpd_limit=100000)
        created: Final = submit_batch(gateway, model, key, rows=3, words=40)
        assert created.status_code == 200, created.text
        assert _creates(wire) == 1


def test_a_key_enqueued_token_limit_refuses_a_batch_larger_than_the_allowance(gateway: Gateway) -> None:
    """A key has ``metadata.batch_enqueued_token_limit: 10`` (set by the admin). A batch of 3 rows needs far more than
    10 tokens, so the create is refused 429 with ``Batch enqueued token limit exceeded for api_key: <hash>. Batch
    requires <n> tokens but only 10 enqueued tokens remaining out of 10 enqueued token limit. Tokens free up as
    running batches complete or are cancelled.`` and ``rate_limit_type: tokens``."""
    with wire_server(BatchProvider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], metadata={"batch_enqueued_token_limit": 10})
        refused: Final = submit_batch(gateway, model, key, rows=3)
        assert refused.status_code == 429, refused.text
        assert re.fullmatch(
            rf"Batch enqueued token limit exceeded for api_key: {hashed(key)}\. Batch requires \d+ tokens but only "
            r"10 enqueued tokens remaining out of 10 enqueued token limit\. Tokens free up as running batches "
            r"complete or are cancelled\.",
            error_message(refused),
        ), refused.text
        assert refused.headers["rate_limit_type"] == "tokens", dict(refused.headers)
        assert _creates(wire) == 0


def test_a_team_enqueued_token_limit_names_the_team(gateway: Gateway) -> None:
    """The team, not the key, carries ``metadata.batch_enqueued_token_limit: 10``. The refusal names ``team:
    <team_id>`` (``resolve_batch_enqueued_token_scopes`` reads ``team_metadata``), so two keys of the team share one
    allowance."""
    with wire_server(BatchProvider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        team: Final = scenario.team(metadata={"batch_enqueued_token_limit": 10}, models=[model])
        refused: Final = submit_batch(gateway, model, scenario.key(team_id=team, models=[model]), rows=3)
        assert refused.status_code == 429, refused.text
        assert error_message(refused).startswith(f"Batch enqueued token limit exceeded for team: {team}. "), (
            refused.text
        )
        assert _creates(wire) == 0


def test_an_enqueued_allowance_is_held_until_the_batch_is_cancelled(gateway: Gateway) -> None:
    """A key's allowance fits one 3 row batch but not two. The first create is served and Redis holds a positive
    ``batch_enqueued_tokens:api_key:<hash>``; the second is refused for ``api_key``. After ``POST
    /v1/batches/<id>/cancel`` answers ``cancelled`` the reservation is refunded (``BATCH_ENQUEUED_REFUND_STATUSES``),
    the counter key is deleted when it reaches zero, and a third create is served."""
    with wire_server(BatchProvider()) as wire, gateway.scenario() as scenario, shared_redis() as redis:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        probe_key: Final = scenario.key(models=[model], metadata={"batch_enqueued_token_limit": 1})
        needed: Final = re.search(
            r"Batch requires (\d+) tokens", error_message(submit_batch(gateway, model, probe_key, rows=3))
        )
        assert needed is not None
        allowance: Final = int(needed.group(1)) * 3 // 2
        key: Final = scenario.key(models=[model], metadata={"batch_enqueued_token_limit": allowance})
        counter: Final = f"batch_enqueued_tokens:api_key:{hashed(key)}"
        first: Final = submit_batch(gateway, model, key, rows=3)
        assert first.status_code == 200, first.text
        assert (redis_integer(redis, counter) or 0) > 0
        second: Final = submit_batch(gateway, model, key, rows=3)
        assert second.status_code == 429, second.text
        assert f"for api_key: {hashed(key)}." in error_message(second), second.text
        cancelled: Final = gateway.request("POST", f"/v1/batches/{batch_id(first)}/cancel", key=key)
        assert cancelled.status_code == 200, cancelled.text
        eventually(lambda: redis.get(counter), lambda value: value is None, seconds=20)
        assert submit_batch(gateway, model, key, rows=3).status_code == 200
        assert _creates(wire) == 2


def test_a_key_with_an_enqueued_allowance_is_not_charged_its_per_minute_rpm_for_a_batch(gateway: Gateway) -> None:
    """A key has ``rpm_limit: 3`` and a large ``batch_enqueued_token_limit``. A 10 row batch is accepted: when a key
    or team enqueued scope applies, the per-minute descriptors are skipped entirely (batch_rate_limiter.py:1463, the
    remaining check runs only ``if not key_and_team_scopes``). The same batch without the allowance would be refused
    for ``Batch contains 10 requests``."""
    with wire_server(BatchProvider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], rpm_limit=3, metadata={"batch_enqueued_token_limit": 10_000_000})
        created: Final = submit_batch(gateway, model, key, rows=10)
        assert created.status_code == 200, created.text
        plain: Final = scenario.key(models=[model], rpm_limit=3)
        refused: Final = submit_batch(gateway, model, plain, rows=10)
        assert refused.status_code == 429, refused.text
        assert "Batch contains 10 requests" in error_message(refused), refused.text
        assert _creates(wire) == 1


def test_a_project_model_enqueued_limit_only_limits_the_named_model(gateway: Gateway) -> None:
    """A project's ``model_batch_enqueued_token_limit`` is 10 for model A only. A key of that project is refused a 3
    row batch on A with ``Batch enqueued token limit exceeded for project_model: <project>:<A>.`` and is served the
    same batch on model B (#51608, ``resolve_project_model_batch_enqueued_token_scopes``). UNVERIFIED: that a
    virtual key created with ``project_id`` carries ``project_metadata`` to the batch limiter as the custom auth test
    in ``spend/`` does."""
    with wire_server(BatchProvider()) as wire, gateway.scenario() as scenario:
        limited: Final = scenario.model(api_base=f"{wire.url}/v1")
        other: Final = scenario.model(api_base=f"{wire.url}/v1")
        team: Final = scenario.team(models=[limited, other])
        project: Final = scenario.project(team, metadata={"model_batch_enqueued_token_limit": {limited: 10}})
        key: Final = scenario.key(team_id=team, project_id=project, models=[limited, other])
        refused: Final = submit_batch(gateway, limited, key, rows=3)
        assert refused.status_code == 429, refused.text
        assert error_message(refused).startswith(
            f"Batch enqueued token limit exceeded for project_model: {project}:{limited}. "
        ), refused.text
        assert submit_batch(gateway, other, key, rows=3).status_code == 200
        assert _creates(wire) == 1
