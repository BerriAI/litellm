import uuid
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway, eventually, gateway_from_environment
from integration._support.process import owned_proxy

_BASE_CONFIG: Final = Path(__file__).resolve().parents[1] / "proxy_config.yaml"
_INPUT_COST: Final = 0.001
_OUTPUT_COST: Final = 0.002
_CHARGED: Final = (
    "SELECT k.spend AS key_spend, t.spend AS team_spend, "
    '(SELECT array_agg(s.spend) FROM "LiteLLM_SpendLogs" s WHERE s.api_key = k.token) AS logged '
    'FROM "LiteLLM_VerificationToken" k JOIN "LiteLLM_TeamTable" t ON t.team_id = k.team_id WHERE k.key_alias=%s'
)


@pytest.fixture(scope="module", params=(True, False), ids=("ledger_on", "ledger_off"))
def spend_gateway(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("no-log-spend")
    base: Final = yaml.safe_load(_BASE_CONFIG.read_text())
    config: Final = directory / "no-log-spend.yaml"
    config.write_text(
        yaml.safe_dump(
            {**base, "general_settings": {**base["general_settings"], "use_spend_logs_ledger": request.param}}
        )
    )
    with gateway_from_environment() as shared, owned_proxy(shared, directory, {}, config=config) as proxy:
        yield proxy


def test_no_log_request_is_charged_to_key_team_and_spend_log(spend_gateway: Gateway) -> None:
    with spend_gateway.scenario() as scenario:
        model: Final = scenario.model(input_cost_per_token=_INPUT_COST, output_cost_per_token=_OUTPUT_COST)
        alias: Final = f"integration-no-log-{uuid.uuid4().hex}"
        key: Final = scenario.key(team_id=scenario.team(), key_alias=alias, models=[model])
        response: Final = spend_gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": f"no-log spend {alias}"}], "no-log": True},
            key=key,
        )
        assert response.status_code == 200, response.text
        usage: Final = response.json()["usage"]
        cost: Final = usage["prompt_tokens"] * _INPUT_COST + usage["completion_tokens"] * _OUTPUT_COST
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(cost), response.headers
        charged: Final = eventually(
            lambda: spend_gateway.rows(_CHARGED, (alias,)),
            lambda rows: len(rows) == 1 and float(str(rows[0]["key_spend"])) > 0 and rows[0]["logged"] is not None,
            seconds=70,
        )
        assert float(str(charged[0]["key_spend"])) == pytest.approx(cost), charged
        assert float(str(charged[0]["team_spend"])) == pytest.approx(cost), charged
        assert charged[0]["logged"] == [pytest.approx(cost)], charged


_STORED: Final = (
    "SELECT s.spend AS spend, s.proxy_server_request::text AS request, s.response::text AS response "
    'FROM "LiteLLM_SpendLogs" s JOIN "LiteLLM_VerificationToken" k ON k.token = s.api_key WHERE k.key_alias=%s'
)


@pytest.fixture(scope="module")
def prompt_storing_gateway(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("no-log-prompts")
    base: Final = yaml.safe_load(_BASE_CONFIG.read_text())
    config: Final = directory / "no-log-prompts.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                **base,
                "general_settings": {
                    **base["general_settings"],
                    "use_spend_logs_ledger": False,
                    "store_prompts_in_spend_logs": True,
                },
            }
        )
    )
    with gateway_from_environment() as shared, owned_proxy(shared, directory, {}, config=config) as proxy:
        yield proxy


def _stored_row(gateway: Gateway, key: str, alias: str, model: str, prompt: str, no_log: bool) -> Mapping[str, object]:
    response: Final = gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": prompt}], "no-log": no_log},
        key=key,
    )
    assert response.status_code == 200, response.text
    return eventually(lambda: gateway.rows(_STORED, (alias,)), lambda rows: len(rows) == 1, seconds=70)[0]


def test_no_log_request_spend_log_keeps_spend_and_drops_prompt_and_response(prompt_storing_gateway: Gateway) -> None:
    with prompt_storing_gateway.scenario() as scenario:
        model: Final = scenario.model(input_cost_per_token=_INPUT_COST, output_cost_per_token=_OUTPUT_COST)
        marker: Final = f"prompt-marker-{uuid.uuid4().hex}"
        aliases: Final = {no_log: f"integration-no-log-{no_log}-{uuid.uuid4().hex}" for no_log in (False, True)}
        stored: Final = {
            no_log: _stored_row(
                prompt_storing_gateway,
                scenario.key(key_alias=alias, models=[model]),
                alias,
                model,
                f"{marker} {no_log}",
                no_log,
            )
            for no_log, alias in aliases.items()
        }
        assert marker in str(stored[False]["request"]), stored[False]
        assert str(stored[False]["response"]) != "{}", stored[False]
        assert float(str(stored[True]["spend"])) == pytest.approx(float(str(stored[False]["spend"]))), stored
        assert stored[True]["spend"] and float(str(stored[True]["spend"])) > 0, stored[True]
        assert (stored[True]["request"], stored[True]["response"]) == ("{}", "{}"), stored[True]
