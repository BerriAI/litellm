import csv
import io
import os
import signal
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from typing import Final

import httpx
import openai
import pytest
from integration._support.client import Gateway, Scenario, eventually, object_value, string_value
from integration._support.database import read_rows
from integration._support.process import group_members, owned_proxy, owned_proxy_process


def _export_range() -> dict[str, str]:
    today: Final = datetime.now(timezone.utc)
    return {
        "start_date": (today - timedelta(days=1)).strftime("%Y-%m-%d"),
        "end_date": (today + timedelta(days=1)).strftime("%Y-%m-%d"),
        "timezone": "0",
    }


def _team_with_three_keys(
    gateway: Gateway, scenario: Scenario, model: str
) -> tuple[str, tuple[str, ...], tuple[str, ...], dict[str, float]]:
    team: Final = scenario.team(models=[model])
    keys: Final = tuple(scenario.key(team_id=team, models=[model]) for _ in range(3))
    digests: Final = tuple(sha256(key.encode()).hexdigest() for key in keys)
    for key in keys:
        reply: Final = gateway.chat(model, key=key, text=f"team export {uuid.uuid4().hex}")
        assert reply["usage"]["total_tokens"] == 40, reply
    daily: Final = eventually(
        lambda: read_rows('SELECT api_key, spend FROM "LiteLLM_DailyTeamSpend" WHERE team_id=%s', (team,)),
        lambda values: len({row["api_key"] for row in values}) == 3,
        seconds=70,
    )
    spend_by_key: Final = {row["api_key"]: float(row["spend"]) for row in daily}
    return team, keys, digests, spend_by_key


def _export_json(gateway: Gateway, **params: str) -> httpx.Response:
    return gateway.request("GET", "/team/daily/activity/export", params={**_export_range(), **params})


def test_team_activity_export_returns_every_key_beyond_the_top_n_cap(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        team: Final = scenario.team(models=[model])
        keys: Final = tuple(scenario.key(team_id=team, models=[model]) for _ in range(3))
        digests: Final = tuple(sha256(key.encode()).hexdigest() for key in keys)
        for key in keys:
            reply: Final = gateway.chat(model, key=key, text=f"team export {uuid.uuid4().hex}")
            assert reply["usage"]["total_tokens"] == 40, reply
        daily: Final = eventually(
            lambda: read_rows('SELECT api_key, spend FROM "LiteLLM_DailyTeamSpend" WHERE team_id=%s', (team,)),
            lambda values: len({row["api_key"] for row in values}) == 3,
            seconds=70,
        )
        spend_by_key: Final = {row["api_key"]: float(row["spend"]) for row in daily}
        response: Final = gateway.request(
            "GET",
            "/team/daily/activity/export",
            params={
                **_export_range(),
                "team_id": team,
                "export_type": "daily_with_keys",
                "format": "json",
            },
        )
        assert response.status_code == 200, response.text
        body: Final = object_value(response.json())
        rows: Final = tuple(object_value(row) for row in body["data"])
        assert sorted(string_value(row["api_key"]) for row in rows) == sorted(digests), response.text
        for row in rows:
            assert row["team_id"] == team, response.text
            assert float(row["spend"]) == pytest.approx(spend_by_key[string_value(row["api_key"])]), response.text
        metadata: Final = object_value(body["metadata"])
        assert (
            metadata["export_type"],
            metadata["team_ids"],
            metadata["total_api_requests"],
            metadata["total_successful_requests"],
            metadata["total_failed_requests"],
        ) == ("daily_with_keys", [team], 3, 3, 0), response.text
        assert float(metadata["total_spend"]) == pytest.approx(sum(spend_by_key.values())), response.text


def test_team_activity_export_csv_downloads_every_key(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        team: Final = scenario.team(models=[model])
        keys: Final = tuple(scenario.key(team_id=team, models=[model]) for _ in range(3))
        digests: Final = tuple(sha256(key.encode()).hexdigest() for key in keys)
        for key in keys:
            reply: Final = gateway.chat(model, key=key, text=f"team export {uuid.uuid4().hex}")
            assert reply["usage"]["total_tokens"] == 40, reply
        daily: Final = eventually(
            lambda: read_rows('SELECT api_key, spend FROM "LiteLLM_DailyTeamSpend" WHERE team_id=%s', (team,)),
            lambda values: len({row["api_key"] for row in values}) == 3,
            seconds=70,
        )
        spend_by_key: Final = {row["api_key"]: float(row["spend"]) for row in daily}
        response: Final = gateway.request(
            "GET",
            "/team/daily/activity/export",
            params={
                **_export_range(),
                "team_id": team,
                "export_type": "daily_with_keys",
                "format": "csv",
            },
        )
        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("text/csv"), response.headers
        assert "attachment" in response.headers["content-disposition"], response.headers
        records: Final = tuple(csv.DictReader(io.StringIO(response.text)))
        assert len(records) == 3, response.text
        assert sorted(record["Key ID"] for record in records) == sorted(digests), response.text
        assert sorted(record["Team ID"] for record in records) == [team, team, team], response.text
        for record in records:
            assert record["Spend ($)"] == f"{spend_by_key[record['Key ID']]:.4f}", response.text


def test_team_activity_export_denies_a_member_another_team(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        team_a: Final = scenario.team(models=[model])
        team_b: Final = scenario.team(models=[model])
        member: Final = scenario.user(user_role="internal_user", teams=[team_a])
        member_key: Final = scenario.key(user_id=member, team_id=team_a, models=[model])
        reply: Final = gateway.chat(model, key=member_key, text=f"team export {uuid.uuid4().hex}")
        assert reply["usage"]["total_tokens"] == 40, reply
        daily: Final = eventually(
            lambda: read_rows('SELECT api_key, spend FROM "LiteLLM_DailyTeamSpend" WHERE team_id=%s', (team_a,)),
            lambda values: len(values) == 1,
            seconds=70,
        )
        denied: Final = gateway.request(
            "GET",
            "/team/daily/activity/export",
            params={**_export_range(), "team_id": team_b, "export_type": "daily", "format": "json"},
            key=member_key,
        )
        assert denied.status_code == 404, denied.text
        assert f"User does not belong to Team= {team_b}" in denied.text, denied.text
        allowed: Final = gateway.request(
            "GET",
            "/team/daily/activity/export",
            params={**_export_range(), "team_id": team_a, "export_type": "daily", "format": "json"},
            key=member_key,
        )
        assert allowed.status_code == 200, allowed.text
        rows: Final = tuple(object_value(row) for row in object_value(allowed.json())["data"])
        assert len(rows) == 1, allowed.text
        assert rows[0]["team_id"] == team_a, allowed.text
        assert float(rows[0]["spend"]) == pytest.approx(float(daily[0]["spend"])), allowed.text


def test_export_daily_total_matches_the_capped_aggregated_team_spend(gateway: Gateway, tmp_path: Path) -> None:
    with owned_proxy(gateway, tmp_path, {"USAGE_TOP_API_KEYS_LIMIT": "2"}, workers=2) as candidate:
        with candidate.scenario() as scenario:
            model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
            team, keys, digests, spend_by_key = _team_with_three_keys(candidate, scenario, model)
            aggregated: Final = candidate.request(
                "GET",
                "/team/daily/activity/aggregated",
                params={**_export_range(), "team_ids": team},
            )
            assert aggregated.status_code == 200, aggregated.text
            body: Final = object_value(aggregated.json())
            metadata: Final = object_value(body["metadata"])
            assert metadata["api_key_limit"] == 2, aggregated.text
            assert metadata["total_api_keys"] == 3, aggregated.text
            day: Final = object_value(body["results"][0])
            breakdown: Final = object_value(day["breakdown"])
            assert len(object_value(breakdown["api_keys"])) == 2, aggregated.text
            team_spend: Final = float(
                object_value(object_value(object_value(breakdown["entities"])[team])["metrics"])["spend"]
            )

            response: Final = _export_json(candidate, team_id=team, export_type="daily", format="json")
            assert response.status_code == 200, response.text
            rows: Final = tuple(object_value(row) for row in object_value(response.json())["data"])
            assert len(rows) == 1, response.text
            assert rows[0]["team_id"] == team, response.text
            assert float(rows[0]["spend"]) == pytest.approx(team_spend), response.text
            assert float(rows[0]["spend"]) == pytest.approx(sum(spend_by_key.values())), response.text


def test_export_users_folds_spend_per_user_and_leaves_keyless_keys_unassigned(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        team: Final = scenario.team(models=[model])
        user_a: Final = scenario.user(user_role="internal_user", teams=[team])
        user_b: Final = scenario.user(user_role="internal_user", teams=[team])
        key_a: Final = scenario.key(team_id=team, user_id=user_a, models=[model])
        key_b: Final = scenario.key(team_id=team, user_id=user_b, models=[model])
        key_none: Final = scenario.key(team_id=team, models=[model])
        for key in (key_a, key_b, key_none):
            reply: Final = gateway.chat(model, key=key, text=f"team export {uuid.uuid4().hex}")
            assert reply["usage"]["total_tokens"] == 40, reply
        daily: Final = eventually(
            lambda: read_rows('SELECT api_key, spend FROM "LiteLLM_DailyTeamSpend" WHERE team_id=%s', (team,)),
            lambda values: len({row["api_key"] for row in values}) == 3,
            seconds=70,
        )
        spend_by_key: Final = {row["api_key"]: float(row["spend"]) for row in daily}
        response: Final = _export_json(gateway, team_id=team, export_type="daily_with_users", format="json")
        assert response.status_code == 200, response.text
        rows: Final = tuple(object_value(row) for row in object_value(response.json())["data"])
        by_user: Final = {row["user_id"]: row for row in rows}
        assert by_user[user_a]["spend"] == pytest.approx(spend_by_key[sha256(key_a.encode()).hexdigest()]), (
            response.text
        )
        assert by_user[user_b]["spend"] == pytest.approx(spend_by_key[sha256(key_b.encode()).hexdigest()]), (
            response.text
        )
        assert None in by_user, response.text
        assert by_user[None]["spend"] == pytest.approx(spend_by_key[sha256(key_none.encode()).hexdigest()]), (
            response.text
        )
        metadata: Final = object_value(object_value(response.json())["metadata"])
        assert float(metadata["total_spend"]) == pytest.approx(sum(spend_by_key.values())), response.text


def test_export_models_reports_one_row_per_model_with_matching_spend(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        upstream_a: Final = f"openai/export-{uuid.uuid4().hex}"
        upstream_b: Final = f"openai/export-{uuid.uuid4().hex}"
        model_a: Final = scenario.model(model=upstream_a, input_cost_per_token=0.001, output_cost_per_token=0.002)
        model_b: Final = scenario.model(model=upstream_b, input_cost_per_token=0.0005, output_cost_per_token=0.001)
        upstream_models: Final = (upstream_a, upstream_b)
        team: Final = scenario.team(models=[model_a, model_b])
        key: Final = scenario.key(team_id=team, models=[model_a, model_b])
        for model in (model_a, model_b):
            reply: Final = gateway.chat(model, key=key, text=f"team export {uuid.uuid4().hex}")
            assert reply["usage"]["total_tokens"] == 40, reply
        daily: Final = eventually(
            lambda: read_rows('SELECT model, spend FROM "LiteLLM_DailyTeamSpend" WHERE team_id=%s', (team,)),
            lambda values: len({row["model"] for row in values}) == 2,
            seconds=70,
        )
        spend_by_model: Final = {row["model"]: float(row["spend"]) for row in daily}

        response: Final = _export_json(gateway, team_id=team, export_type="daily_with_models", format="json")
        assert response.status_code == 200, response.text
        rows: Final = tuple(object_value(row) for row in object_value(response.json())["data"])
        assert {row["model"] for row in rows} == set(upstream_models), response.text
        for row in rows:
            assert float(row["spend"]) == pytest.approx(spend_by_model[row["model"]]), response.text

        csv_response: Final = _export_json(gateway, team_id=team, export_type="daily_with_models", format="csv")
        assert csv_response.status_code == 200, csv_response.text
        records: Final = tuple(csv.DictReader(io.StringIO(csv_response.text)))
        assert sorted(record["Model"] for record in records) == sorted(upstream_models), csv_response.text


def test_export_without_team_id_returns_only_the_callers_teams(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        team_a: Final = scenario.team(models=[model])
        team_b: Final = scenario.team(models=[model])
        member: Final = scenario.user(user_role="internal_user", teams=[team_a])
        member_key: Final = scenario.key(user_id=member, team_id=team_a, models=[model])
        other_key: Final = scenario.key(team_id=team_b, models=[model])
        reply: Final = gateway.chat(model, key=member_key, text=f"team export {uuid.uuid4().hex}")
        assert reply["usage"]["total_tokens"] == 40, reply
        reply_b: Final = gateway.chat(model, key=other_key, text=f"team export {uuid.uuid4().hex}")
        assert reply_b["usage"]["total_tokens"] == 40, reply_b
        eventually(
            lambda: read_rows('SELECT team_id FROM "LiteLLM_DailyTeamSpend" WHERE team_id=%s', (team_b,)),
            lambda values: len(values) == 1,
            seconds=70,
        )
        eventually(
            lambda: read_rows('SELECT team_id FROM "LiteLLM_DailyTeamSpend" WHERE team_id=%s', (team_a,)),
            lambda values: len(values) == 1,
            seconds=70,
        )

        response: Final = gateway.request(
            "GET",
            "/team/daily/activity/export",
            params={**_export_range(), "export_type": "daily", "format": "json"},
            key=member_key,
        )
        assert response.status_code == 200, response.text
        rows: Final = tuple(object_value(row) for row in object_value(response.json())["data"])
        assert len(rows) == 1, response.text
        assert rows[0]["team_id"] == team_a, response.text


def test_export_rejects_requests_without_a_valid_key(gateway: Gateway) -> None:
    params: Final = {**_export_range(), "export_type": "daily", "format": "json"}
    anonymous: Final = gateway.client.get("/team/daily/activity/export", params=params)
    assert anonymous.status_code == 401, anonymous.text
    garbage: Final = gateway.request("GET", "/team/daily/activity/export", params=params, key="sk-nope")
    assert garbage.status_code == 401, garbage.text


def test_export_rejects_bad_parameters(gateway: Gateway) -> None:
    weekly: Final = _export_json(gateway, export_type="weekly", format="json")
    assert weekly.status_code == 422, weekly.text
    xml: Final = _export_json(gateway, export_type="daily", format="xml")
    assert xml.status_code == 422, xml.text
    no_dates: Final = gateway.request(
        "GET", "/team/daily/activity/export", params={"export_type": "daily", "format": "json"}
    )
    assert no_dates.status_code == 400, no_dates.text
    assert "start_date and end_date" in no_dates.text, no_dates.text
    reversed_range: Final = gateway.request(
        "GET",
        "/team/daily/activity/export",
        params={"start_date": "2026-09-25", "end_date": "2026-09-23", "export_type": "daily", "format": "json"},
    )
    assert reversed_range.status_code == 400, reversed_range.text
    assert "end_date must be on or after start_date" in reversed_range.text, reversed_range.text
    bad_date: Final = gateway.request(
        "GET",
        "/team/daily/activity/export",
        params={"start_date": "2026-13-40", "end_date": "2026-12-31", "export_type": "daily", "format": "json"},
    )
    assert bad_date.status_code == 400, bad_date.text
    assert "valid YYYY-MM-DD" in bad_date.text, bad_date.text


def test_export_of_a_team_without_spend_returns_empty(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        team: Final = scenario.team(models=[model])
        fresh: Final = _export_json(gateway, team_id=team, export_type="daily", format="json")
        assert fresh.status_code == 200, fresh.text
        body: Final = object_value(fresh.json())
        assert body["data"] == [], fresh.text
        assert float(object_value(body["metadata"])["total_spend"]) == 0, fresh.text
    unknown: Final = _export_json(gateway, team_id=str(uuid.uuid4()), export_type="daily", format="json")
    assert unknown.status_code == 200, unknown.text
    unknown_body: Final = object_value(unknown.json())
    assert unknown_body["data"] == [], unknown.text
    assert float(object_value(unknown_body["metadata"])["total_spend"]) == 0, unknown.text


def test_export_csv_is_deterministic_and_omits_flat_cost_without_ptu(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        _team_with_three_keys(gateway, scenario, model)
        params: Final = {**_export_range(), "export_type": "daily_with_keys", "format": "csv"}
        first: Final = gateway.request("GET", "/team/daily/activity/export", params=params)
        second: Final = gateway.request("GET", "/team/daily/activity/export", params=params)
        assert first.status_code == 200 and second.status_code == 200, first.text
        assert first.text == second.text, "daily_with_keys csv is not byte-identical across calls"
        header: Final = first.text.splitlines()[0]
        assert "Flat Cost" not in header and "Total Cost" not in header, header


def test_export_csv_escapes_formula_like_key_aliases(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        team: Final = scenario.team(models=[model])
        alias: Final = f'=HYPERLINK("http://x.{uuid.uuid4().hex}","x")'
        keys: Final = (
            scenario.key(team_id=team, models=[model], key_alias=alias),
            scenario.key(team_id=team, models=[model]),
        )
        for key in keys:
            reply: Final = gateway.chat(model, key=key, text=f"team export {uuid.uuid4().hex}")
            assert reply["usage"]["total_tokens"] == 40, reply
        digests: Final = tuple(sha256(key.encode()).hexdigest() for key in keys)
        eventually(
            lambda: read_rows('SELECT api_key FROM "LiteLLM_DailyTeamSpend" WHERE team_id=%s', (team,)),
            lambda values: len({row["api_key"] for row in values}) == 2,
            seconds=70,
        )
        response: Final = _export_json(gateway, team_id=team, export_type="daily_with_keys", format="csv")
        assert response.status_code == 200, response.text
        records: Final = {record["Key ID"]: record for record in csv.DictReader(io.StringIO(response.text))}
        assert records[digests[0]]["Key Alias"] == "'" + alias, response.text
        assert records[digests[1]]["Key Alias"] == "-", response.text


def test_aggregated_route_keeps_the_top_n_key_cap(gateway: Gateway, tmp_path: Path) -> None:
    with owned_proxy(gateway, tmp_path, {"USAGE_TOP_API_KEYS_LIMIT": "2"}, workers=2) as candidate:
        with candidate.scenario() as scenario:
            model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
            team, keys, digests, spend_by_key = _team_with_three_keys(candidate, scenario, model)
            response: Final = candidate.request(
                "GET",
                "/team/daily/activity/aggregated",
                params={**_export_range(), "team_ids": team},
            )
            assert response.status_code == 200, response.text
            body: Final = object_value(response.json())
            metadata: Final = object_value(body["metadata"])
            assert metadata["api_key_limit"] == 2, response.text
            assert metadata["total_api_keys"] == 3, response.text
            breakdown: Final = object_value(object_value(body["results"][0])["breakdown"])
            assert len(object_value(breakdown["api_keys"])) == 2, response.text


def test_paginated_team_daily_activity_still_lists_the_team(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        team, keys, digests, spend_by_key = _team_with_three_keys(gateway, scenario, model)
        response: Final = gateway.request(
            "GET",
            "/team/daily/activity",
            params={
                "team_ids": team,
                "start_date": _export_range()["start_date"],
                "end_date": _export_range()["end_date"],
            },
        )
        assert response.status_code == 200, response.text
        results: Final = object_value(response.json())["results"]
        assert isinstance(results, list), response.text
        days: Final = tuple(
            object_value(day)
            for day in results
            if team in object_value(object_value(object_value(day)["breakdown"])["entities"])
        )
        assert len(days) == 1, response.text
        entity: Final = object_value(object_value(object_value(days[0]["breakdown"])["entities"])[team])
        assert float(object_value(entity["metrics"])["spend"]) == pytest.approx(sum(spend_by_key.values())), (
            response.text
        )


def test_openai_sdk_chat_still_lands_one_spend_log(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        team: Final = scenario.team(models=[model])
        key: Final = scenario.key(team_id=team, models=[model])
        client: Final = openai.OpenAI(base_url=f"{gateway.client.base_url}/v1", api_key=key, max_retries=0)
        reply: Final = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": f"sdk {uuid.uuid4().hex}"}], stream=False
        )
        rows: Final = eventually(
            lambda: read_rows('SELECT request_id FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (reply.id,)),
            lambda values: len(values) == 1,
            seconds=70,
        )
        assert len(rows) == 1 and rows[0]["request_id"] == reply.id, rows


def test_export_and_chat_burst_survives_worker_kill(gateway: Gateway, tmp_path: Path) -> None:
    with owned_proxy_process(gateway, tmp_path, {"USAGE_TOP_API_KEYS_LIMIT": "2"}, workers=2) as owned:
        candidate: Final = owned.gateway
        with candidate.scenario() as scenario:
            model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
            team, keys, digests, spend_by_key = _team_with_three_keys(candidate, scenario, model)

            workers: Final = eventually(
                lambda: tuple(member for member in group_members(owned.process.pid) if member.pid != owned.process.pid),
                lambda members: len(members) >= 2,
                seconds=30,
            )
            assert len(workers) >= 2, workers

            params: Final = {
                **_export_range(),
                "team_id": team,
                "export_type": "daily_with_keys",
                "format": "json",
            }

            def burst(tag: str) -> tuple[tuple[httpx.Response, ...], tuple[httpx.Response, ...]]:
                with ThreadPoolExecutor(max_workers=30) as pool:
                    futures: Final = tuple(
                        (
                            pool.submit(
                                candidate.request,
                                "POST",
                                "/v1/chat/completions",
                                {
                                    "model": model,
                                    "messages": [{"role": "user", "content": f"{tag}-{index}-{uuid.uuid4().hex}"}],
                                },
                                key=keys[index % 3],
                            )
                            if index % 2 == 0
                            else pool.submit(candidate.request, "GET", "/team/daily/activity/export", params=params)
                        )
                        for index in range(30)
                    )
                    results: Final = tuple(future.result() for future in futures)
                return results[0::2], results[1::2]

            chat_a, export_a = burst("bursta")
            assert all(response.status_code == 200 for response in chat_a), [r.text for r in chat_a]
            assert all(response.status_code == 200 for response in export_a), [r.text for r in export_a]

            victim: Final = workers[0]
            os.kill(victim.pid, signal.SIGKILL)

            chat_b, export_b = burst("burstb")
            all_chats: Final = chat_a + chat_b
            all_exports: Final = export_a + export_b
            assert all(response.status_code == 200 for response in all_chats), [
                (r.status_code, r.text) for r in all_chats
            ]
            for response in all_exports:
                assert response.status_code == 200, response.text
                returned: Final = {string_value(row["api_key"]) for row in object_value(response.json())["data"]}
                assert returned == set(digests), response.text
            chat_ids: Final = tuple(string_value(object_value(r.json())["id"]) for r in all_chats)
            assert len(set(chat_ids)) == 30
            id_slots: Final = ", ".join("%s" for _ in chat_ids)
            rows: Final = eventually(
                lambda: read_rows(
                    f'SELECT request_id, COUNT(*)::int AS n FROM "LiteLLM_SpendLogs" WHERE request_id IN ({id_slots}) GROUP BY request_id',
                    chat_ids,
                ),
                lambda values: len(values) == 30,
                seconds=70,
            )
            assert all(row["n"] == 1 for row in rows), rows
