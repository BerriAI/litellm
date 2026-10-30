import uuid
from collections.abc import Mapping
from datetime import date, datetime, time, timedelta
from hashlib import sha256
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, eventually, object_value
from integration._support.database import read_rows
from pydantic import BaseModel


class ModelDetail(BaseModel):
    model: str
    total_cost: float
    total_input_tokens: float
    total_output_tokens: float
    team_id: str | None = None


class ReportRow(BaseModel):
    api_key: str
    total_cost: float
    total_input_tokens: float
    total_output_tokens: float
    model_details: list[ModelDetail]


def _chat(gateway: Gateway, model: str, key: str) -> None:
    response: Final = gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "max_tokens": 20, "messages": [{"role": "user", "content": f"report {uuid.uuid4().hex}"}]},
        key=key,
    )
    assert response.status_code == 200, response.text


def _report(
    gateway: Gateway, path: str, *, key: str | None = None, params: Mapping[str, str] | None = None
) -> httpx.Response:
    return gateway.request("GET", path, key=key, params=params)


def _rows(
    gateway: Gateway, path: str, *, key: str | None = None, params: Mapping[str, str] | None = None
) -> list[ReportRow]:
    response: Final = _report(gateway, path, key=key, params=params)
    assert response.status_code == 200, response.text
    return [ReportRow.model_validate(row) for row in response.json()]


def _logged(scope_column: str, scope_value: str, start: datetime, end: datetime) -> list[dict[str, object]]:
    return read_rows(
        f'SELECT api_key, model, spend, prompt_tokens, completion_tokens, team_id FROM "LiteLLM_SpendLogs" '
        f'WHERE "startTime" >= %s AND "startTime" < %s AND {scope_column} = %s',
        (start.isoformat(), end.isoformat(), scope_value),
    )


def _logged_keys(keys: tuple[str, ...], start: datetime, end: datetime) -> list[dict[str, object]]:
    placeholders: Final = ", ".join("%s" for _ in keys)
    return read_rows(
        'SELECT api_key, model, spend, prompt_tokens, completion_tokens, team_id FROM "LiteLLM_SpendLogs" '
        f'WHERE "startTime" >= %s AND "startTime" < %s AND api_key IN ({placeholders})',
        (start.isoformat(), end.isoformat(), *keys),
    )


def _assert_report_matches_db(report: list[ReportRow], logged: list[dict[str, object]]) -> None:
    assert len(report) == len({row["api_key"] for row in logged}), (report, logged)
    by_key: Final[dict[str, dict[str, list[float]]]] = {}
    for row in logged:
        bucket: Final = by_key.setdefault(str(row["api_key"]), {})
        model_bucket: Final = bucket.setdefault(str(row["model"]), [0.0, 0.0, 0.0])
        model_bucket[0] += float(str(row["spend"]))
        model_bucket[1] += int(str(row["prompt_tokens"]))
        model_bucket[2] += int(str(row["completion_tokens"]))
    for entry in report:
        assert entry.api_key in by_key, (entry, logged)
        models: Final = by_key[entry.api_key]
        assert entry.total_cost == pytest.approx(sum(v[0] for v in models.values())), entry
        assert entry.total_input_tokens == pytest.approx(sum(v[1] for v in models.values())), entry
        assert entry.total_output_tokens == pytest.approx(sum(v[2] for v in models.values())), entry
        details: Final = {detail.model: detail for detail in entry.model_details}
        assert set(details) == set(models), (entry, logged)
        for model, detail in details.items():
            cost, prompt, completion = models[model]
            assert detail.total_cost == pytest.approx(cost), detail
            assert detail.total_input_tokens == pytest.approx(prompt), detail
            assert detail.total_output_tokens == pytest.approx(completion), detail


def test_scoped_spend_reports_match_sql_and_enforce_caller_scope(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        org1: Final = scenario.organization()
        org2: Final = scenario.organization()
        team1: Final = scenario.team(organization_id=org1)
        team2: Final = scenario.team(organization_id=org2)
        user1: Final = scenario.member(team1, role="user")
        user2: Final = scenario.member(team2, role="user")
        org_admin: Final = scenario.org_member(org1, "org_admin")
        k1: Final = scenario.key(user_id=user1, team_id=team1, models=[model])
        k2: Final = scenario.key(user_id=user2, team_id=team2, models=[model])
        direct_org_key: Final = scenario.key(organization_id=org2, models=[model])
        org_admin_key: Final = scenario.key(user_id=org_admin, organization_id=org1, models=[model])
        org_admin_info: Final = object_value(gateway.get("/key/info", {"key": org_admin_key})["info"])
        assert org_admin_info["organization_id"] == org1, org_admin_info
        _chat(gateway, model, k1)
        _chat(gateway, model, k2)
        _chat(gateway, model, direct_org_key)
        k1_hash: Final = sha256(k1.encode()).hexdigest()
        k2_hash: Final = sha256(k2.encode()).hexdigest()
        direct_org_key_hash: Final = sha256(direct_org_key.encode()).hexdigest()
        logged: Final = eventually(
            lambda: read_rows(
                "SELECT api_key, to_char(\"startTime\", 'YYYY-MM-DD') AS day "
                'FROM "LiteLLM_SpendLogs" WHERE api_key IN (%s, %s, %s)',
                (k1_hash, k2_hash, direct_org_key_hash),
            ),
            lambda rows: len(rows) == 3,
            seconds=30,
        )
        dates: Final = {date.fromisoformat(str(row["day"])) for row in logged}
        start_date: Final[date] = min(dates)
        end_date: Final[date] = max(dates)
        start: Final = datetime.combine(start_date, time.min)
        end: Final = datetime.combine(end_date, time.min) + timedelta(days=1)
        window: Final = {"start_date": start_date.isoformat(), "end_date": end_date.isoformat()}

        _assert_report_matches_db(
            _rows(gateway, "/key/spend/report", key=k1, params=window), _logged("api_key", k1_hash, start, end)
        )
        _assert_report_matches_db(
            _rows(gateway, "/user/spend/report", key=k1, params=window), _logged('"user"', user1, start, end)
        )
        _assert_report_matches_db(
            _rows(gateway, "/team/spend/report", key=k1, params=window), _logged("team_id", team1, start, end)
        )
        org1_report: Final = _rows(gateway, "/organization/spend/report", key=org_admin_key, params=window)
        _assert_report_matches_db(org1_report, _logged_keys((k1_hash,), start, end))
        refused_org2: Final = _report(
            gateway,
            "/organization/spend/report",
            key=org_admin_key,
            params={**window, "organization_id": org2},
        )
        assert refused_org2.status_code == 403, refused_org2.text
        assert refused_org2.json() == {"detail": "You do not have access to this organization"}, refused_org2.text

        own_org: Final = _report(gateway, "/organization/spend/report", key=k1, params=window)
        assert own_org.status_code == 403, own_org.text
        assert own_org.json() == {"detail": "You do not have access to this organization"}, own_org.text

        assert _rows(gateway, "/key/spend/report", key=k1, params={**window, "api_key": k1}) == _rows(
            gateway, "/key/spend/report", key=k1, params={**window, "api_key": k1_hash}
        )

        for path, param, value, detail in (
            (
                "/key/spend/report",
                "api_key",
                k2_hash,
                "Not authorized to view spend for a api_key other than your own",
            ),
            (
                "/user/spend/report",
                "internal_user_id",
                user2,
                "Not authorized to view spend for a internal_user_id other than your own",
            ),
            (
                "/team/spend/report",
                "team_id",
                team2,
                "Not authorized to view spend for a team_id other than your own",
            ),
            ("/organization/spend/report", "organization_id", org2, "You do not have access to this organization"),
        ):
            refused: Final = _report(gateway, path, key=k1, params={**window, param: value})
            assert refused.status_code == 403, refused.text
            assert refused.json() == {"detail": detail}, refused.text

        _assert_report_matches_db(
            _rows(gateway, "/key/spend/report", params={**window, "api_key": k2_hash}),
            _logged("api_key", k2_hash, start, end),
        )
        _assert_report_matches_db(
            _rows(gateway, "/user/spend/report", params={**window, "internal_user_id": user2}),
            _logged('"user"', user2, start, end),
        )
        _assert_report_matches_db(
            _rows(gateway, "/team/spend/report", params={**window, "team_id": team2}),
            _logged("team_id", team2, start, end),
        )
        org_report: Final = _rows(gateway, "/organization/spend/report", params={**window, "organization_id": org2})
        _assert_report_matches_db(org_report, _logged_keys((k2_hash, direct_org_key_hash), start, end))
        assert {detail.team_id for entry in org_report for detail in entry.model_details} == {team2, ""}, org_report


@pytest.mark.parametrize(
    "path",
    (
        "/key/spend/report",
        "/user/spend/report",
        "/team/spend/report",
        "/organization/spend/report",
    ),
)
@pytest.mark.parametrize(
    ("date_kind", "params", "detail"),
    (
        ("missing", None, "Please provide start_date and end_date"),
        (
            "malformed",
            {"start_date": "2026-13-45", "end_date": "2026-01-01"},
            "start_date and end_date must be in YYYY-MM-DD format",
        ),
    ),
)
def test_spend_report_rejects_missing_and_malformed_dates(
    gateway: Gateway, path: str, date_kind: str, params: Mapping[str, str] | None, detail: str
) -> None:
    response: Final = _report(gateway, path, params=params)
    assert response.status_code == 400, response.text
    assert response.json() == {"detail": detail}, response.text
