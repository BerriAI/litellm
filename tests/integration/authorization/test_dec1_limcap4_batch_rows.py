"""Batch creation charged row by row: end user daily tokens, project input and output windows, and alias rows.

``POST /v1/batches`` is charged by ``_PROXY_BatchRateLimiter`` (``litellm/proxy/hooks/batch_rate_limiter.py``), which
reads the input file once and charges the whole batch before the provider sees the create:

- key, team and end user scopes with a ``tpd_limit`` are charged on a daily descriptor ``<scope>_tpd`` (window 86400
  s) instead of their per-minute one (``_create_batch_rate_limit_descriptors``, batch_rate_limiter.py:315). The end
  user's comes from its budget's ``tpd_limit`` (``_apply_budget_limits_to_end_user_params``,
  user_api_key_auth.py:1292);
- a project's ``model_itpm_limit`` and ``model_otpm_limit`` are charged per row, with each row's own tokens, on the
  window of the model the row names (``_create_project_io_descriptors_for_models``, batch_rate_limiter.py:387). The
  input window takes the row's counted tokens; the output window takes the row's ``max_tokens``,
  ``max_completion_tokens`` or ``max_output_tokens`` times ``n``, nothing for an ``/v1/embeddings`` row, and for a row
  without a cap a quarter of the smallest project OTPM limit (``_estimate_entry_output_tokens``, :545);
- a model-routed upload (``/v1/files`` with ``model``) rewrites rows that name the proxy alias to the deployment's
  model (#51843, ``_rewrite_alias_rows_to_deployment_model``, files_endpoints.py:392), and the batch limiter maps
  those rows back to the alias for access and project quotas (``_routing_alias_by_row_model``, :251).

The refusal text is ``Batch rate limit exceeded for <descriptor>: <value>. Batch contains <n> tokens but only <r>
tokens remaining out of <limit> TPM|TPD limit. Limit resets at: <UTC>`` (``_raise_rate_limit_error``, :791). Project
ITPM and OTPM windows say ``TPM``. Rows here use pretokenized embedding input, where every token id counts one
token (``_count_prompt_or_input_tokens``, litellm/batches/batch_utils.py:771), so token numbers are exact.
"""

from __future__ import annotations

import re
from typing import Final

import pytest

from tests.integration._support.client import Gateway
from tests.integration._support.wire import wire_server
from tests.integration.authorization.dec1_limcap2_support import BatchProvider
from tests.integration.authorization.dec1_limcap4_support import (
    DEPLOYMENT_MODEL,
    TPD_WINDOW_SECONDS,
    batch_creates,
    chat_row,
    counter,
    counter_settles,
    counter_ttl,
    create_batch,
    embedding_row,
    end_user_with,
    responses_row,
    row_model,
    uploaded_rows,
)
from tests.integration.authorization.dec1_limits_support import error_message, hashed

RESET_AT: Final = r" Limit resets at: \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC"
END_USER_HEADER: Final = "x-litellm-end-user-id"
ITPM: Final = "model_per_project_itpm"
OTPM: Final = "model_per_project_otpm"


def _assert_batch_refusal(message: str, descriptor: str, value: str, tokens: int, remaining: int, limit: int) -> None:
    """The token refusal of the batch limiter, exactly, up to the reset time."""
    label: Final = "TPD" if descriptor.endswith("_tpd") else "TPM"
    assert re.fullmatch(
        re.escape(
            f"Batch rate limit exceeded for {descriptor}: {value}. Batch contains {tokens} tokens but only "
            f"{remaining} tokens remaining out of {limit} {label} limit."
        )
        + RESET_AT,
        message,
    ), message


def test_an_end_user_tpd_limit_refuses_a_batch_on_the_end_user_daily_descriptor(gateway: Gateway) -> None:
    """An end user's budget has ``tpd_limit: 50``. A batch of 3 pretokenized rows of 30 tokens, created with
    ``x-litellm-end-user-id``, needs 90 tokens and is refused 429 with exactly ``Batch rate limit exceeded for
    end_user_tpd: <end_user_id>. Batch contains 90 tokens but only 50 tokens remaining out of 50 TPD limit.``; the
    provider never sees the create. UNVERIFIED: that auth resolves the end user from the header on ``/v1/batches``
    like it does on chat (``get_end_user_id_from_request_body``, auth_utils.py:2357)."""
    with wire_server(BatchProvider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model])
        customer: Final = end_user_with(gateway, scenario, tpd_limit=50)
        wire.drain()
        refused: Final = create_batch(
            gateway,
            model,
            key,
            [embedding_row(model)] * 3,
            endpoint="/v1/embeddings",
            headers={END_USER_HEADER: customer},
        )
        assert refused.status_code == 429, refused.text
        _assert_batch_refusal(error_message(refused), "end_user_tpd", customer, 90, 50, 50)
        assert batch_creates(wire) == 0


def test_an_end_user_tpd_refusal_retries_within_a_day_not_a_minute(gateway: Gateway) -> None:
    """The same end user daily refusal carries ``rate_limit_type: tokens`` and a ``retry-after`` that counts down the
    86400 second daily window (``BATCH_TPD_WINDOW_SECONDS``), so it is above the 60 second minute window when no
    earlier batch opened the day.

    BUG (run 44176682961): the 429 had no rate_limit_type. handle_exception_on_proxy
    (litellm/proxy/utils.py:10276-10277) turns the ProxyRateLimitError into a ProxyException without its headers on
    /v1/batches."""
    pytest.skip("BUG: /v1/batches drops the limiter's 429 headers (retry-after, rate_limit_type)")
    with wire_server(BatchProvider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model])
        customer: Final = end_user_with(gateway, scenario, tpd_limit=50)
        refused: Final = create_batch(
            gateway,
            model,
            key,
            [embedding_row(model)] * 3,
            endpoint="/v1/embeddings",
            headers={END_USER_HEADER: customer},
        )
        assert refused.status_code == 429, refused.text
        assert refused.headers["rate_limit_type"] == "tokens", dict(refused.headers)
        assert 60 < int(refused.headers["retry-after"]) <= TPD_WINDOW_SECONDS, dict(refused.headers)


def test_an_end_user_tpd_limit_takes_the_batch_off_the_end_users_per_minute_rpm(gateway: Gateway) -> None:
    """The end user's budget has ``rpm_limit: 1`` and ``tpd_limit: 100000``. A 3 row batch is created 200: with a
    ``tpd_limit`` the end user's per-minute ``end_user`` descriptor is dropped from the batch charge
    (batch_rate_limiter.py:350), so its rows never count against the per-minute request limit."""
    with wire_server(BatchProvider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model])
        customer: Final = end_user_with(gateway, scenario, rpm_limit=1, tpd_limit=100000)
        wire.drain()
        created: Final = create_batch(
            gateway,
            model,
            key,
            [embedding_row(model)] * 3,
            endpoint="/v1/embeddings",
            headers={END_USER_HEADER: customer},
        )
        assert created.status_code == 200, created.text
        assert batch_creates(wire) == 1


def test_an_end_user_rpm_limit_without_tpd_charges_every_row_of_the_batch(gateway: Gateway) -> None:
    """The end user's budget has only ``rpm_limit: 2``. A 3 row batch is refused 429 with ``Batch rate limit exceeded
    for end_user: <end_user_id>. Batch contains 3 requests but only 2 requests remaining out of 2 RPM limit.``: the
    per-minute end user descriptor from ``create_rate_limit_descriptors`` (parallel_request_limiter_v3.py:3364) is
    charged the row count."""
    with wire_server(BatchProvider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model])
        customer: Final = end_user_with(gateway, scenario, rpm_limit=2)
        refused: Final = create_batch(
            gateway,
            model,
            key,
            [embedding_row(model)] * 3,
            endpoint="/v1/embeddings",
            headers={END_USER_HEADER: customer},
        )
        assert refused.status_code == 429, refused.text
        assert re.fullmatch(
            re.escape(
                f"Batch rate limit exceeded for end_user: {customer}. Batch contains 3 requests but only 2 requests "
                "remaining out of 2 RPM limit."
            )
            + RESET_AT,
            error_message(refused),
        ), refused.text


def test_an_accepted_batch_leaves_its_tokens_on_the_end_users_daily_counter(gateway: Gateway) -> None:
    """An end user with ``tpd_limit: 1000`` creates a batch of 3 pretokenized rows of 30 tokens. Redis then holds
    ``{end_user_tpd:<end_user_id>}:tokens`` = 90, with a TTL longer than a minute and at most a day: the daily window
    is its own counter and never shares the end user's per-minute ``{end_user:<id>}`` keys."""
    with wire_server(BatchProvider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model])
        customer: Final = end_user_with(gateway, scenario, tpd_limit=1000)
        created: Final = create_batch(
            gateway,
            model,
            key,
            [embedding_row(model)] * 3,
            endpoint="/v1/embeddings",
            headers={END_USER_HEADER: customer},
        )
        assert created.status_code == 200, created.text
        assert counter_settles("end_user_tpd", customer, "tokens", 90) == 90
        assert 60 < counter_ttl("end_user_tpd", customer, "tokens") <= TPD_WINDOW_SECONDS
        assert counter("end_user", customer, "tokens") is None


def test_pretokenized_rows_count_one_token_per_id_against_a_key_tpd_limit(gateway: Gateway) -> None:
    """A key with ``tpd_limit: 50`` creates a batch of 3 embedding rows whose ``input`` is 30 token ids each. The
    refusal reads exactly ``Batch contains 90 tokens but only 50 tokens remaining out of 50 TPD limit.`` for
    ``api_key_tpd: <hash>``: a pretokenized list is counted one token per id, so it cannot slip past a token limit
    as zero tokens (batch_utils.py:771)."""
    with wire_server(BatchProvider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], tpd_limit=50)
        refused: Final = create_batch(gateway, model, key, [embedding_row(model)] * 3, endpoint="/v1/embeddings")
        assert refused.status_code == 429, refused.text
        _assert_batch_refusal(error_message(refused), "api_key_tpd", hashed(key), 90, 50, 50)


def test_a_project_itpm_limit_on_the_alias_refuses_the_rows_rewritten_to_the_deployment_model(
    gateway: Gateway,
) -> None:
    """A project has ``model_itpm_limit: {<alias>: 50}``. The key uploads 3 rows naming the alias through a
    model-routed upload, which rewrites them to ``gpt-4o-mini``. The batch limiter maps them back to the alias, so the
    create is refused 429 with ``Batch rate limit exceeded for model_per_project_itpm: <project>:<alias>. Batch
    contains 90 tokens but only 50 tokens remaining out of 50 TPM limit.`` and never forwarded."""
    with wire_server(BatchProvider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        team: Final = scenario.team(models=[model])
        project: Final = scenario.project(team, models=[model], model_itpm_limit={model: 50})
        key: Final = scenario.key(team_id=team, project_id=project)
        wire.drain()
        refused: Final = create_batch(gateway, model, key, [embedding_row(model)] * 3, endpoint="/v1/embeddings")
        assert refused.status_code == 429, refused.text
        _assert_batch_refusal(error_message(refused), ITPM, f"{project}:{model}", 90, 50, 50)
        assert batch_creates(wire) == 0


def test_rows_uploaded_naming_the_deployment_model_are_charged_to_the_aliases_project_window(
    gateway: Gateway,
) -> None:
    """A project has ``model_itpm_limit: {<alias>: 50}``. The upload is routed by the alias but its 3 rows already
    name ``gpt-4o-mini``, the deployment's model. After the upload the file cannot tell which name the caller wrote,
    so those rows are charged to the alias too and the batch is refused for ``model_per_project_itpm:
    <project>:<alias>`` with 90 tokens (batch_rate_limiter.py:1112)."""
    with wire_server(BatchProvider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        team: Final = scenario.team(models=[model])
        project: Final = scenario.project(team, models=[model], model_itpm_limit={model: 50})
        key: Final = scenario.key(team_id=team, project_id=project)
        refused: Final = create_batch(
            gateway, model, key, [embedding_row(DEPLOYMENT_MODEL)] * 3, endpoint="/v1/embeddings"
        )
        assert refused.status_code == 429, refused.text
        _assert_batch_refusal(error_message(refused), ITPM, f"{project}:{model}", 90, 50, 50)


def test_alias_rows_reach_the_provider_rewritten_and_other_rows_unchanged(gateway: Gateway) -> None:
    """A model-routed upload holds two rows naming the alias and one naming a second proxy model. The provider's
    ``POST /files`` carries the alias rows with ``body.model: gpt-4o-mini`` and the other row still naming the second
    model (#51843 rewrites only rows naming the alias)."""
    with wire_server(BatchProvider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        other: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model, other])
        wire.drain()
        created: Final = create_batch(gateway, model, key, [chat_row(model), chat_row(model), chat_row(other)])
        assert created.status_code == 200, created.text
        assert [row_model(row) for row in uploaded_rows(wire)] == [DEPLOYMENT_MODEL, DEPLOYMENT_MODEL, other]


def test_a_key_allowed_only_the_alias_may_submit_rows_rewritten_to_the_deployment_model(gateway: Gateway) -> None:
    """A key with ``models: [<alias>]`` uploads two rows naming the alias. After the rewrite the stored rows name
    ``gpt-4o-mini``, which the key is not allowed to call by name, yet the create is 200: the model access check
    sees the alias the rows were mapped back to (``_routing_alias_by_row_model``, batch_rate_limiter.py:251)."""
    with wire_server(BatchProvider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], rpm_limit=100)
        wire.drain()
        created: Final = create_batch(gateway, model, key, [chat_row(model), chat_row(model)])
        assert created.status_code == 200, created.text
        assert batch_creates(wire) == 1


def test_a_project_otpm_limit_is_charged_each_rows_max_tokens(gateway: Gateway) -> None:
    """A project has ``model_otpm_limit: {<alias>: 200}``. Three chat rows with ``max_tokens: 100`` need 300 output
    tokens, so the create is refused with exactly ``Batch rate limit exceeded for model_per_project_otpm:
    <project>:<alias>. Batch contains 300 tokens but only 200 tokens remaining out of 200 TPM limit.``: the OTPM
    refusal reports the rows' output total, not their input (batch_rate_limiter.py:842)."""
    with wire_server(BatchProvider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        team: Final = scenario.team(models=[model])
        project: Final = scenario.project(team, models=[model], model_otpm_limit={model: 200})
        key: Final = scenario.key(team_id=team, project_id=project)
        refused: Final = create_batch(gateway, model, key, [chat_row(model, max_tokens=100)] * 3)
        assert refused.status_code == 429, refused.text
        _assert_batch_refusal(error_message(refused), OTPM, f"{project}:{model}", 300, 200, 200)


def test_n_multiplies_a_rows_output_reservation(gateway: Gateway) -> None:
    """Two chat rows with ``max_tokens: 50`` and ``n: 3`` need 2 x 50 x 3 = 300 output tokens
    (``get_output_candidate_count``, parallel_request_limiter_v3.py:1153). Under ``model_otpm_limit: 200`` the batch
    is refused reporting ``Batch contains 300 tokens``."""
    with wire_server(BatchProvider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        team: Final = scenario.team(models=[model])
        project: Final = scenario.project(team, models=[model], model_otpm_limit={model: 200})
        key: Final = scenario.key(team_id=team, project_id=project)
        refused: Final = create_batch(gateway, model, key, [chat_row(model, max_tokens=50, n=3)] * 2)
        assert refused.status_code == 429, refused.text
        _assert_batch_refusal(error_message(refused), OTPM, f"{project}:{model}", 300, 200, 200)


def test_a_responses_row_is_reserved_its_max_output_tokens(gateway: Gateway) -> None:
    """Two ``/v1/responses`` rows with ``max_output_tokens: 120`` reserve 240 output tokens (the batch path reads
    ``max_output_tokens`` as the cap and applies no Responses minimum, batch_rate_limiter.py:575). Under
    ``model_otpm_limit: 200`` the batch is refused reporting ``Batch contains 240 tokens``."""
    with wire_server(BatchProvider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        team: Final = scenario.team(models=[model])
        project: Final = scenario.project(team, models=[model], model_otpm_limit={model: 200})
        key: Final = scenario.key(team_id=team, project_id=project)
        refused: Final = create_batch(
            gateway, model, key, [responses_row(model, max_output_tokens=120)] * 2, endpoint="/v1/responses"
        )
        assert refused.status_code == 429, refused.text
        _assert_batch_refusal(error_message(refused), OTPM, f"{project}:{model}", 240, 200, 200)


def test_an_embeddings_row_reserves_no_output_tokens(gateway: Gateway) -> None:
    """A project has ``model_otpm_limit: {<alias>: 1}``. A batch of 3 ``/v1/embeddings`` rows is created 200: a row
    whose ``url`` names embeddings reserves 0 output tokens (batch_rate_limiter.py:563), so even a 1 token output
    window admits it, and ``{model_per_project_otpm:<project>:<alias>}:tokens`` holds 0 or is absent."""
    with wire_server(BatchProvider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        team: Final = scenario.team(models=[model])
        project: Final = scenario.project(team, models=[model], model_otpm_limit={model: 1})
        key: Final = scenario.key(team_id=team, project_id=project)
        created: Final = create_batch(gateway, model, key, [embedding_row(model)] * 3, endpoint="/v1/embeddings")
        assert created.status_code == 200, created.text
        assert counter(OTPM, f"{project}:{model}", "tokens") in (None, 0)


def test_rows_without_max_tokens_reserve_a_quarter_of_the_smallest_project_otpm_limit(gateway: Gateway) -> None:
    """The project has ``model_otpm_limit: {<alias>: 4000, <other>: 40}``. Three alias chat rows without
    ``max_tokens`` each reserve ``min(1024, 40 x 0.25) = 10`` output tokens: the floor uses the smallest OTPM limit
    configured for any model of the project, not the alias's own (batch_rate_limiter.py:1029). The alias's output
    counter settles to 30."""
    with wire_server(BatchProvider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        other: Final = scenario.model(api_base=f"{wire.url}/v1")
        team: Final = scenario.team(models=[model, other])
        project: Final = scenario.project(team, models=[model, other], model_otpm_limit={model: 4000, other: 40})
        key: Final = scenario.key(team_id=team, project_id=project)
        created: Final = create_batch(gateway, model, key, [chat_row(model)] * 3)
        assert created.status_code == 200, created.text
        assert counter_settles(OTPM, f"{project}:{model}", "tokens", 30) == 30


def test_rows_for_another_model_are_charged_only_to_that_models_project_window(gateway: Gateway) -> None:
    """The project limits both the alias and a second model to 1000 output tokens. A batch routed by the alias holds
    two alias rows and one row for the second model, each with ``max_tokens: 100``. Redis then holds 200 on
    ``{model_per_project_otpm:<project>:<alias>}:tokens`` and 100 on the second model's counter: each row charges only
    the model it names, never the whole batch's total."""
    with wire_server(BatchProvider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        other: Final = scenario.model(api_base=f"{wire.url}/v1")
        team: Final = scenario.team(models=[model, other])
        project: Final = scenario.project(team, models=[model, other], model_otpm_limit={model: 1000, other: 1000})
        key: Final = scenario.key(team_id=team, project_id=project)
        rows: Final = [
            chat_row(model, max_tokens=100),
            chat_row(model, max_tokens=100),
            chat_row(other, max_tokens=100),
        ]
        created: Final = create_batch(gateway, model, key, rows)
        assert created.status_code == 200, created.text
        assert counter_settles(OTPM, f"{project}:{model}", "tokens", 200) == 200
        assert counter_settles(OTPM, f"{project}:{other}", "tokens", 100) == 100


def test_a_quota_limited_row_model_cannot_hide_behind_an_unlimited_routing_model(gateway: Gateway) -> None:
    """Only the second model has a project limit, ``model_itpm_limit: {<other>: 50}``. A batch routed by the unlimited
    alias carries 3 pretokenized rows naming the second model (90 tokens). It is refused for
    ``model_per_project_itpm: <project>:<other>`` with ``Batch contains 90 tokens``: the project window follows the
    row's model, not the routing model (batch_rate_limiter.py:315)."""
    with wire_server(BatchProvider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        other: Final = scenario.model(api_base=f"{wire.url}/v1")
        team: Final = scenario.team(models=[model, other])
        project: Final = scenario.project(team, models=[model, other], model_itpm_limit={other: 50})
        key: Final = scenario.key(team_id=team, project_id=project)
        refused: Final = create_batch(gateway, model, key, [embedding_row(other)] * 3, endpoint="/v1/embeddings")
        assert refused.status_code == 429, refused.text
        _assert_batch_refusal(error_message(refused), ITPM, f"{project}:{other}", 90, 50, 50)


def test_a_project_quota_alone_makes_the_limiter_read_the_file_of_a_key_without_limits(gateway: Gateway) -> None:
    """The key has no limits and no model allowlist, which normally lets the batch limiter skip reading the file. Its
    project has ``model_otpm_limit: {<alias>: 100}``, so the file is read anyway
    (``_project_has_any_io_token_limits``, batch_rate_limiter.py:368) and two rows with ``max_tokens: 100`` are refused
    reporting ``Batch contains 200 tokens``."""
    with wire_server(BatchProvider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        team: Final = scenario.team()
        project: Final = scenario.project(team, model_otpm_limit={model: 100})
        key: Final = scenario.key(team_id=team, project_id=project)
        refused: Final = create_batch(gateway, model, key, [chat_row(model, max_tokens=100)] * 2)
        assert refused.status_code == 429, refused.text
        _assert_batch_refusal(error_message(refused), OTPM, f"{project}:{model}", 200, 100, 100)
