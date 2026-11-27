import json
import uuid
from collections.abc import Iterator, Mapping
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway, eventually, gateway_from_environment
from integration._support.process import owned_proxy
from integration._support.rejection_probe import REJECT_HEADER, REJECTION_DETAIL
from integration._support.wire import Reply, Wire, wire_server
from pydantic import JsonValue

pytestmark: Final = pytest.mark.timeout(240)

MODEL: Final = "rejected-request-model"
REDACT: Final = {"x-litellm-enable-message-redaction": "true"}
DECISIONS_QUESTIONS: Final[JsonValue] = [
    {"type": "predicate", "name": "billing", "instructions": "Is this about billing?"}
]
SYSTEMONE_QUESTIONS: Final[JsonValue] = {
    "billing": {
        "type": "noul",
        "instructions": "Is this a billing request?",
        "criteria": {"true": "Concerns a charge.", "false": "Not billing."},
    }
}


@pytest.fixture(scope="module")
def rejecting_proxy(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Gateway, Wire, Wire]]:
    tmp_path: Final = tmp_path_factory.mktemp("rejected-request-caller-input")
    with (
        wire_server(lambda request: Reply(status=500)) as provider,
        wire_server(lambda request: Reply()) as sink,
        gateway_from_environment() as rig,
    ):
        config: Final = tmp_path / "rejected-request.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "model_list": [
                        {
                            "model_name": MODEL,
                            "litellm_params": {
                                "model": "openai/gpt-4o-mini",
                                "api_base": provider.url + "/v1",
                                "api_key": "synthetic-provider-key",
                            },
                        }
                    ],
                    "litellm_settings": {"callbacks": ["integration._support.rejection_probe.probe"]},
                    "general_settings": {
                        "master_key": "os.environ/LITELLM_MASTER_KEY",
                        "database_url": "os.environ/DATABASE_URL",
                        "disable_model_info_refresh": True,
                    },
                }
            )
        )
        with owned_proxy(
            rig, tmp_path, {"REJECTION_PROBE_URL": sink.url, "STORE_MODEL_IN_DB": "False"}, config=config, workers=2
        ) as proxy:
            yield proxy, provider, sink


def _rejected(
    proxy: Gateway,
    sink: Wire,
    route: str,
    rejection: str,
    body: Mapping[str, JsonValue],
    headers: Mapping[str, str] | None = None,
) -> dict[str, JsonValue]:
    sink.drain()
    if rejection == "unknown_key":
        response: Final = proxy.request(
            "POST", route, {"model": MODEL, **body}, key="sk-" + uuid.uuid4().hex, headers=headers
        )
        assert response.status_code == 401, response.text
    else:
        assert rejection == "pre_call_hook", rejection
        response = proxy.request(
            "POST", route, {"model": MODEL, **body}, headers={REJECT_HEADER: "1", **(headers or {})}
        )
        assert response.status_code == 429, response.text
        assert REJECTION_DETAIL in response.text, response.text
    (received,) = eventually(sink.drain, bool, seconds=20)
    return json.loads(received.body)


REJECTIONS: Final = ("unknown_key", "pre_call_hook")


@pytest.mark.parametrize("rejection", REJECTIONS)
@pytest.mark.parametrize(
    ("route", "call_type", "fields"),
    [
        ("/v1/decisions", "adecisions", {"input": "CANARY", "questions": DECISIONS_QUESTIONS}),
        ("/decisions", "adecisions", {"input": "CANARY", "questions": DECISIONS_QUESTIONS}),
        ("/v1/systemone", "asystemone", {"state": {"message": "CANARY"}, "questions": SYSTEMONE_QUESTIONS}),
        ("/systemone", "asystemone", {"state": {"message": "CANARY"}, "questions": SYSTEMONE_QUESTIONS}),
    ],
    ids=["v1_decisions", "decisions", "v1_systemone", "systemone"],
)
def test_rejected_decisions_request_logs_the_fields_the_caller_sent(
    rejecting_proxy: tuple[Gateway, Wire, Wire],
    route: str,
    call_type: str,
    fields: Mapping[str, JsonValue],
    rejection: str,
) -> None:
    proxy, provider, sink = rejecting_proxy
    canary: Final = "prompt-" + uuid.uuid4().hex
    sent: Final = json.loads(json.dumps(fields).replace("CANARY", canary))

    logged: Final = _rejected(proxy, sink, route, rejection, sent)

    standard: Final = logged["standard"]
    assert standard["status"] == "failure", standard
    assert standard["call_type"] == call_type, standard
    (message,) = standard["messages"]
    assert message["role"] == "user", message
    assert json.loads(message["content"]) == sent, message
    assert provider.drain() == ()


@pytest.mark.parametrize("rejection", REJECTIONS)
@pytest.mark.parametrize("route", ["/v1/responses", "/responses"])
@pytest.mark.parametrize(
    "shape",
    [
        "CANARY",
        ["first turn", "CANARY"],
        [{"role": "user", "content": "CANARY"}],
    ],
    ids=["text", "texts", "messages"],
)
def test_rejected_responses_request_logs_the_input_the_caller_sent(
    rejecting_proxy: tuple[Gateway, Wire, Wire], route: str, shape: JsonValue, rejection: str
) -> None:
    proxy, provider, sink = rejecting_proxy
    canary: Final = "prompt-" + uuid.uuid4().hex
    sent: Final = json.loads(json.dumps(shape).replace("CANARY", canary))

    logged: Final = _rejected(proxy, sink, route, rejection, {"input": sent})

    assert logged["input"] == sent, logged
    assert logged["standard"]["call_type"] == "aresponses", logged
    assert canary in json.dumps(logged["standard"]["messages"]), logged
    assert provider.drain() == ()


@pytest.mark.parametrize("rejection", REJECTIONS)
@pytest.mark.parametrize(
    ("route", "call_type", "fields"),
    [
        ("/v1/decisions", "adecisions", {"input": "CANARY", "questions": DECISIONS_QUESTIONS}),
        ("/v1/systemone", "asystemone", {"state": {"message": "CANARY"}, "questions": SYSTEMONE_QUESTIONS}),
        ("/v1/responses", "aresponses", {"input": "CANARY"}),
    ],
    ids=["decisions", "systemone", "responses"],
)
def test_rejected_request_log_keeps_the_caller_input_out_when_the_caller_asks_for_redaction(
    rejecting_proxy: tuple[Gateway, Wire, Wire],
    route: str,
    call_type: str,
    fields: Mapping[str, JsonValue],
    rejection: str,
) -> None:
    proxy, _, sink = rejecting_proxy
    canary: Final = "prompt-" + uuid.uuid4().hex
    sent: Final = json.loads(json.dumps(fields).replace("CANARY", canary))

    logged: Final = _rejected(proxy, sink, route, rejection, sent, REDACT)

    assert logged["standard"]["status"] == "failure", logged
    assert logged["standard"]["call_type"] == call_type, logged
    assert canary not in json.dumps(logged), logged
