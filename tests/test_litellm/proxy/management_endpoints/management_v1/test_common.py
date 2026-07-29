from typing import Optional

from fastapi import Depends, FastAPI, Query, Request
from fastapi.testclient import TestClient

from litellm.proxy.management_endpoints.management_v1.common import (
    ManagementProblem,
    problem_response,
    reject_unknown_query_params,
)


def _team_scope(team_id: Optional[str] = Query(default=None)) -> Optional[str]:
    return team_id


app = FastAPI()


@app.exception_handler(ManagementProblem)
async def _handle(request: Request, exc: ManagementProblem):
    return problem_response(exc.problem)


@app.get("/things", dependencies=[Depends(reject_unknown_query_params)])
def _things(page: int = Query(default=1), _scope: Optional[str] = Depends(_team_scope)) -> dict:
    return {"ok": True}


client = TestClient(app)


def test_accepts_query_params_declared_on_the_route_and_nested_dependencies():
    """`team_id` is declared on a nested dependency, not the route signature, so a
    shallow scan of the route's own params would wrongly reject it."""
    assert client.get("/things?page=2&team_id=t1").status_code == 200


def test_rejects_unknown_param_and_reports_nested_declared_params_as_allowed():
    response = client.get("/things?bogus=1")

    assert response.status_code == 400
    body = response.json()
    assert "bogus" in body["detail"]
    assert {"page", "team_id"}.issubset(set(body["allowed"]))
