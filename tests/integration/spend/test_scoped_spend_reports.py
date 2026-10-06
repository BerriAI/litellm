import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from hashlib import sha256
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, eventually
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


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


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


def _logged(scope_column: str, scope_value: str) -> list[dict[str, object]]:
    return read_rows(
        f'SELECT api_key, model, spend, prompt_tokens, completion_tokens, team_id FROM "LiteLLM_SpendLogs" '
        f"WHERE \"startTime\" >= (CURRENT_DATE AT TIME ZONE 'UTC') AND {scope_column} = %s",
        (scope_value,),
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
    window: Final = {"start_date": _today(), "end_date": _today()}
    with gateway.scenario() as scenario:
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        org1: Final = scenario.organization()
        org2: Final = scenario.organization()
        team1: Final = scenario.team(organization_id=org1)
        team2: Final = scenario.team(organization_id=org2)
        user1: Final = scenario.member(team1, role="user")
        user2: Final = scenario.member(team2, role="user")
        k1: Final = scenario.key(user_id=user1, team_id=team1, models=[model])
        k2: Final = scenario.key(user_id=user2, team_id=team2, models=[model])
        _chat(gateway, model, k1)
        _chat(gateway, model, k2)
        k1_hash: Final = sha256(k1.encode()).hexdigest()
        k2_hash: Final = sha256(k2.encode()).hexdigest()
        for column, value in (("api_key", k1_hash), ("api_key", k2_hash)):
            eventually(
                lambda column=column, value=value: _logged(column, value), lambda rows: len(rows) == 1, seconds=70
            )

        # k1's own scopes, no filter: key, user, team reports return only its rows
        _assert_report_matches_db(
            _rows(gateway, "/key/spend/report", key=k1, params=window), _logged("api_key", k1_hash)
        )
        _assert_report_matches_db(_rows(gateway, "/user/spend/report", key=k1, params=window), _logged('"user"', user1))
        _assert_report_matches_db(
            _rows(gateway, "/team/spend/report", key=k1, params=window), _logged("team_id", team1)
        )
        # a key in a team under org1 has no org scope of its own: own-org reads are also refused
        own_org: Final = _report(gateway, "/organization/spend/report", key=k1, params=window)
        assert own_org.status_code == 403, own_org.text
        assert own_org.json() == {"detail": "You do not have access to this organization"}, own_org.text

        # raw key works like its hash
        assert _rows(gateway, "/key/spend/report", key=k1, params={**window, "api_key": k1}) == _rows(
            gateway, "/key/spend/report", key=k1, params={**window, "api_key": k1_hash}
        )

        # k1 cannot read k2's scopes
        for path, param, value in (
            ("/key/spend/report", "api_key", k2_hash),
            ("/user/spend/report", "internal_user_id", user2),
            ("/team/spend/report", "team_id", team2),
            ("/organization/spend/report", "organization_id", org2),
        ):
            refused: Final = _report(gateway, path, key=k1, params={**window, param: value})
            assert refused.status_code == 403, (path, refused.text)

        # master may read any scope with matching totals
        _assert_report_matches_db(
            _rows(gateway, "/key/spend/report", params={**window, "api_key": k2_hash}), _logged("api_key", k2_hash)
        )
        _assert_report_matches_db(
            _rows(gateway, "/user/spend/report", params={**window, "internal_user_id": user2}),
            _logged('"user"', user2),
        )
        _assert_report_matches_db(
            _rows(gateway, "/team/spend/report", params={**window, "team_id": team2}), _logged("team_id", team2)
        )
        org_report: Final = _rows(gateway, "/organization/spend/report", params={**window, "organization_id": org2})
        _assert_report_matches_db(org_report, _logged("team_id", team2))
        assert {detail.team_id for entry in org_report for detail in entry.model_details} == {team2}, org_report


def test_spend_report_rejects_missing_and_malformed_dates(gateway: Gateway) -> None:
    missing: Final = _report(gateway, "/key/spend/report")
    assert missing.status_code == 400, missing.text
    malformed: Final = _report(gateway, "/key/spend/report", params={"start_date": "2026-13-45", "end_date": _today()})
    assert malformed.status_code == 400, malformed.text
