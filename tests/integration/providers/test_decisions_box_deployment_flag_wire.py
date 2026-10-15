import json
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from queue import SimpleQueue
from types import MappingProxyType
from typing import Final, Literal

import httpx
import pytest
from integration._support.client import Gateway, Scenario, eventually, object_value, string_value
from integration._support.database import read_rows
from integration._support.process import configless_proxy_process, graceful_stop_seconds
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

import litellm
from litellm.constants import PROXY_CONFIG_RELOAD_INTERVAL_SECONDS

_API_KEY: Final = "synthetic-decisions-key"
_IMAGES_HEADER: Final = "TypeSafe-Danger-Zone-Images"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_JSON_LIST: Final = TypeAdapter(list[JsonValue])
_INPUT: Final = "Photo description: a phone screen with a diagonal crack from the top-left corner to the middle."
_BOX: Final[dict[str, JsonValue]] = {"top": 0.1, "bottom": 0.4, "left": 0.2, "right": 0.6}
_PREDICATE: Final[dict[str, JsonValue]] = {
    "type": "predicate",
    "name": "cracked",
    "instructions": "Is the screen cracked?",
}
_BOX_QUESTION: Final[dict[str, JsonValue]] = {"type": "box", "name": "crack", "instructions": "Where is the crack?"}
_QUESTIONS: Final[tuple[dict[str, JsonValue], ...]] = (_PREDICATE, _BOX_QUESTION)
_PREDICATE_ANSWER: Final[dict[str, JsonValue]] = {"type": "predicate", "name": "cracked", "probability": 0.9}
_BOX_ANSWER: Final[dict[str, JsonValue]] = {"type": "box", "name": "crack", "box": _BOX, "confidence": 0.7}
_BOX_REFUSAL: Final[dict[str, JsonValue]] = {"type": "refusal", "name": "crack"}
_SYSTEM_ONE_QUESTIONS: Final[dict[str, JsonValue]] = {
    "cracked": {"type": "noul", "instructions": "Is the screen cracked?"},
    "crack": {"type": "box", "instructions": "Where is the crack?"},
}
_SYSTEM_ONE_ANSWERS: Final[dict[str, JsonValue]] = {
    "cracked": {"type": "noul", "noul": 0.9},
    "crack": {"type": "box", "box": _BOX, "confidence": 0.7},
}
_USAGE: Final[dict[str, JsonValue]] = {"input_tokens": 41, "output_tokens": 2}
_OPENAI_USAGE: Final[dict[str, JsonValue]] = {
    "input_tokens": 41,
    "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
    "output_tokens": 2,
    "output_tokens_details": {"reasoning_tokens": 0},
    "total_tokens": 43,
}
_REFUSED_BOX: Final = "does not support parameters: ['box questions']"
_SPEND_QUERY: Final = (
    'SELECT status, call_type, custom_llm_provider, model_group, model_id FROM "LiteLLM_SpendLogs" '
    "WHERE request_id = %s"
)
_OPENAI_FORMAT_ROUTES: Final = ("/v1/decisions", "/decisions")
_SYSTEM_ONE_ROUTES: Final = ("/v1/systemone", "/systemone")

Envelope = Literal["plain", "result", "ai_decide"]


def _systemone_question(question: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    kind: Final = "box" if question["type"] == "box" else "noul"
    return {"type": kind, "instructions": question["instructions"]}


def _systemone_answer(question: JsonValue, *, probability_key: str) -> JsonValue:
    if object_value(question)["type"] == "box":
        return {"type": "box", "box": _BOX, "confidence": 0.7}
    return {"type": "noul", probability_key: 0.9}


def _openai_answer(question: JsonValue) -> JsonValue:
    fields: Final = object_value(question)
    template: Final = _BOX_ANSWER if fields["type"] == "box" else _PREDICATE_ANSWER
    return {**template, "name": fields["name"]}


def _expected_answers(questions: Sequence[Mapping[str, JsonValue]]) -> list[JsonValue]:
    return [_openai_answer(dict(question)) for question in questions]


@dataclass(frozen=True, slots=True)
class _Member:
    provider: str
    model: str
    path: str
    body_model: str | None
    api_key: str | None = _API_KEY
    speaks_openai: bool = False
    envelope: Envelope = "plain"
    headers: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))

    def deployment(
        self, scenario: Scenario, api_base: str, model_info: Mapping[str, JsonValue] | None, **extra: JsonValue
    ) -> str:
        header_params: Final[dict[str, JsonValue]] = {"extra_headers": dict(self.headers)} if self.headers else {}
        return scenario.model(
            model=self.model,
            api_base=api_base,
            api_key=self.api_key,
            model_info=model_info,
            **header_params,
            **extra,
        )

    def upstream_body(self, questions: Sequence[Mapping[str, JsonValue]]) -> dict[str, JsonValue]:
        if self.speaks_openai:
            return {"model": self.body_model, "input": _INPUT, "questions": [dict(question) for question in questions]}
        keyed: Final[dict[str, JsonValue]] = {
            string_value(question["name"]): _systemone_question(question) for question in questions
        }
        model_field: Final[dict[str, JsonValue]] = {} if self.body_model is None else {"model": self.body_model}
        return {**model_field, "state": _INPUT, "questions": keyed}

    def reply(self, request: Request) -> Reply:
        body: Final = _JSON_OBJECT.validate_json(request.body)
        if self.speaks_openai:
            questions: Final = _JSON_LIST.validate_python(body["questions"])
            answer: Final = {
                "model": self.body_model,
                "answers": list(map(_openai_answer, questions)),
                "usage": _OPENAI_USAGE,
            }
            return Reply(body=json.dumps(answer).encode())
        probability_key: Final = "probability" if self.envelope == "ai_decide" else "noul"
        answers: Final = {
            key: _systemone_answer(question, probability_key=probability_key)
            for key, question in object_value(body["questions"]).items()
        }
        return Reply(body=json.dumps(self._enveloped(answers)).encode())

    def _enveloped(self, answers: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        plain: Final[dict[str, JsonValue]] = {"model": self.body_model, "answers": dict(answers), "usage": _USAGE}
        if self.envelope == "result":
            return {"result": plain, "success": True}
        if self.envelope == "ai_decide":
            return {"response": {"answers": dict(answers)}, "metadata": {"version": "1.0"}}
        return plain


def _members(tag: str) -> Mapping[str, _Member]:
    name: Final = f"box-{tag}"
    return MappingProxyType(
        {
            "perplexity": _Member("perplexity", f"perplexity/{name}", "/v1/decisions", name),
            "typesafe": _Member(
                "typesafe",
                f"typesafe/{name}",
                "/v1/systemone",
                name,
                headers=MappingProxyType({_IMAGES_HEADER: "true"}),
            ),
            "openrouter": _Member("openrouter", f"openrouter/typesafe/{name}", "/alpha/decisions", f"typesafe/{name}"),
            "fireworks_ai": _Member(
                "fireworks_ai",
                f"fireworks_ai/accounts/fireworks/routers/{name}",
                "/v1/systemone",
                f"accounts/fireworks/routers/{name}",
            ),
            "cloudflare": _Member(
                "cloudflare", f"cloudflare/{name}", f"/ai/run/@cf/cloudflare/{name}", name, envelope="result"
            ),
            "strands_decider": _Member("strands_decider", f"strands_decider/{name}", "/v1/systemone", name, None),
            "databricks": _Member("databricks", f"databricks/{name}", f"/{name}/invocations", name),
            "databricks_ai_decide": _Member(
                "databricks", "databricks/ai_decide", "/api/2.0/ai-functions/ai-decide", None, envelope="ai_decide"
            ),
            "hosted_vllm": _Member("hosted_vllm", f"hosted_vllm/{name}", "/v1/systemone", name, None),
            "laya": _Member("laya", "laya/typed-decisions", "/v1/systemone", "typed-decisions", None),
            "bespoke": _Member(
                "bespoke",
                "bespoke/bespokelabs/Bespoke-Nimble-9B",
                "/v1/systemone",
                "bespokelabs/Bespoke-Nimble-9B",
                None,
            ),
            "azure_ai": _Member("azure_ai", f"azure_ai/{name}", "/providers/microsoft/v1/systemone", name),
            "openai": _Member("openai", f"openai/{name}", "/v1/decisions", name, speaks_openai=True),
        }
    )


_MEMBER_LABELS: Final = tuple(_members("labels"))


def _tag() -> str:
    return uuid.uuid4().hex[:12]


def _perplexity() -> _Member:
    return _members(_tag())["perplexity"]


def _decide(
    gateway: Gateway,
    model: str,
    questions: Sequence[Mapping[str, JsonValue]] = _QUESTIONS,
    *,
    route: str = "/v1/decisions",
    **extra: JsonValue,
) -> httpx.Response:
    body: Final[dict[str, JsonValue]] = {
        "model": model,
        "input": _INPUT,
        "questions": [dict(question) for question in questions],
        "num_retries": 0,
        **extra,
    }
    return gateway.request("POST", route, body)


def _decide_in_system_one_format(gateway: Gateway, model: str, route: str, **extra: JsonValue) -> httpx.Response:
    body: Final[dict[str, JsonValue]] = {
        "model": model,
        "state": _INPUT,
        "questions": _SYSTEM_ONE_QUESTIONS,
        "num_retries": 0,
        **extra,
    }
    return gateway.request("POST", route, body)


def _spend_row(call_id: str) -> dict[str, JsonValue]:
    rows: Final = eventually(lambda: read_rows(_SPEND_QUERY, (call_id,)), lambda found: len(found) == 1, seconds=70)
    return rows[0]


def _assert_refused_box(response: httpx.Response) -> None:
    assert response.status_code == 400, response.text
    assert _REFUSED_BOX in response.text, response.text


def _authorization(member: _Member) -> str:
    return f"Bearer {member.api_key}" if member.api_key else ""


@pytest.mark.parametrize("label", _MEMBER_LABELS)
def test_a_flagged_deployment_of_every_decisions_provider_sends_box_questions_and_returns_the_box(
    gateway: Gateway, label: str
) -> None:
    member: Final = _members(_tag())[label]
    with wire_server(member.reply) as wire, gateway.scenario() as scenario:
        model: Final = member.deployment(scenario, wire.url, {"supports_box_questions": True})
        response: Final = _decide(gateway, model)
        assert response.status_code == 200, response.text
        assert response.json()["answers"] == _expected_answers(_QUESTIONS), response.text
        (call,) = wire.drain()
        assert (call.method, call.target) == ("POST", member.path)
        assert call.headers.get("authorization", "") == _authorization(member)
        assert _JSON_OBJECT.validate_json(call.body) == member.upstream_body(_QUESTIONS)
        assert all(call.headers.get(name.lower()) == value for name, value in member.headers.items()), call.headers
        row: Final = _spend_row(response.headers["x-litellm-call-id"])
        assert (row["status"], row["call_type"], row["custom_llm_provider"], row["model_group"]) == (
            "success",
            "adecisions",
            member.provider,
            model,
        )


@pytest.mark.parametrize("route", (*_OPENAI_FORMAT_ROUTES, *_SYSTEM_ONE_ROUTES))
def test_every_route_answers_box_questions_for_a_flagged_deployment_and_refuses_them_for_an_unflagged_one(
    gateway: Gateway, route: str
) -> None:
    flagged_member: Final = _perplexity()
    plain_member: Final = _perplexity()
    send: Final = (
        (lambda model: _decide(gateway, model, route=route))
        if route in _OPENAI_FORMAT_ROUTES
        else (lambda model: _decide_in_system_one_format(gateway, model, route))
    )
    with wire_server(flagged_member.reply) as wire, gateway.scenario() as scenario:
        flagged: Final = flagged_member.deployment(scenario, wire.url, {"supports_box_questions": True})
        plain: Final = plain_member.deployment(scenario, wire.url, None)
        refused: Final = send(plain)
        _assert_refused_box(refused)
        assert wire.drain() == ()
        answered: Final = send(flagged)
        assert answered.status_code == 200, answered.text
        expected: Final[JsonValue] = (
            _expected_answers(_QUESTIONS) if route in _OPENAI_FORMAT_ROUTES else _SYSTEM_ONE_ANSWERS
        )
        assert answered.json()["answers"] == expected, answered.text
        (call,) = wire.drain()
        assert call.target == flagged_member.path
        assert _JSON_OBJECT.validate_json(call.body) == flagged_member.upstream_body(_QUESTIONS)


def test_repeated_identical_box_requests_each_reach_the_upstream_and_each_log_once(gateway: Gateway) -> None:
    member: Final = _perplexity()
    with wire_server(member.reply) as wire, gateway.scenario() as scenario:
        model: Final = member.deployment(scenario, wire.url, {"supports_box_questions": True})
        responses: Final = tuple(_decide(gateway, model, (_BOX_QUESTION,)) for _ in range(2))
        assert [response.status_code for response in responses] == [200, 200], [r.text for r in responses]
        call_ids: Final = tuple(response.headers["x-litellm-call-id"] for response in responses)
        assert len(set(call_ids)) == 2, call_ids
        assert [_JSON_OBJECT.validate_json(call.body) for call in wire.drain()] == [
            member.upstream_body((_BOX_QUESTION,))
        ] * 2
        assert [_spend_row(call_id)["status"] for call_id in call_ids] == ["success", "success"]


async def test_the_sdk_honors_a_model_info_flag_passed_by_its_caller_on_sync_and_async_calls() -> None:
    member: Final = _perplexity()
    with wire_server(member.reply) as wire:
        common: Final[dict[str, object]] = {
            "model": member.model,
            "input": _INPUT,
            "questions": [dict(question) for question in _QUESTIONS],
            "api_base": wire.url,
            "api_key": _API_KEY,
        }
        synchronous: Final = litellm.decisions(**common, model_info={"supports_box_questions": True})
        asynchronous: Final = await litellm.adecisions(**common, model_info={"supports_box_questions": True})
        for response in (synchronous, asynchronous):
            assert response.model_dump(mode="json")["answers"] == _expected_answers(_QUESTIONS)
        with pytest.raises(litellm.UnsupportedParamsError, match=r"\['box questions'\]"):
            await litellm.adecisions(**common)
        with pytest.raises(litellm.UnsupportedParamsError, match=r"\['box questions'\]"):
            litellm.decisions(**common, model_info={"supports_box_questions": "true"})
        assert [_JSON_OBJECT.validate_json(call.body) for call in wire.drain()] == [
            member.upstream_body(_QUESTIONS)
        ] * 2


async def test_a_router_deployment_flagged_false_refuses_boxes_even_when_its_backend_row_says_true() -> None:
    member: Final = _perplexity()
    with wire_server(member.reply) as wire:
        params: Final[dict[str, object]] = {"model": member.model, "api_base": wire.url, "api_key": _API_KEY}
        router: Final = litellm.Router(
            model_list=[
                {"model_name": "box-off", "litellm_params": params, "model_info": {"supports_box_questions": False}},
                {
                    "model_name": "box-on",
                    "litellm_params": params,
                    "model_info": {"supports_box_questions": True, "mode": "evaluation"},
                },
                {"model_name": "box-unset", "litellm_params": params},
            ]
        )
        common: Final[dict[str, object]] = {"input": _INPUT, "questions": [dict(question) for question in _QUESTIONS]}
        with pytest.raises(litellm.UnsupportedParamsError, match=r"\['box questions'\]"):
            await router.adecisions(model="box-off", **common)
        assert wire.drain() == ()
        answered: Final = [await router.adecisions(model=model, **common) for model in ("box-on", "box-unset")]
        assert [response.model_dump(mode="json")["answers"] for response in answered] == [
            _expected_answers(_QUESTIONS)
        ] * 2
        assert [_JSON_OBJECT.validate_json(call.body) for call in wire.drain()] == [
            member.upstream_body(_QUESTIONS)
        ] * 2


@pytest.mark.parametrize(
    "value", ("true", 1, [True], "", "x" * 5000, None), ids=("string", "int", "list", "empty", "5kb", "null")
)
def test_a_non_boolean_deployment_flag_counts_as_no_flag(gateway: Gateway, value: JsonValue) -> None:
    member: Final = _perplexity()
    with wire_server(member.reply) as wire, gateway.scenario() as scenario:
        model: Final = member.deployment(scenario, wire.url, {"supports_box_questions": value})
        _assert_refused_box(_decide(gateway, model))
        assert wire.drain() == ()
        predicate_only: Final = _decide(gateway, model, (_PREDICATE,))
        assert predicate_only.status_code == 200, predicate_only.text
        assert predicate_only.json()["answers"] == [_PREDICATE_ANSWER]
        (call,) = wire.drain()
        assert _JSON_OBJECT.validate_json(call.body) == member.upstream_body((_PREDICATE,))


_BODY_MODEL_INFO: Final[tuple[tuple[str, JsonValue], ...]] = (
    ("flag", {"supports_box_questions": True}),
    ("int", 5),
    ("list", [{"supports_box_questions": True}]),
    ("empty", ""),
    ("5kb", "x" * 5000),
)


@pytest.mark.parametrize("route", ("/v1/decisions", "/v1/systemone"))
@pytest.mark.parametrize(
    "value", [value for _, value in _BODY_MODEL_INFO], ids=[label for label, _ in _BODY_MODEL_INFO]
)
def test_a_body_model_info_never_turns_box_questions_on_and_never_reaches_the_provider(
    gateway: Gateway, route: str, value: JsonValue
) -> None:
    member: Final = _perplexity()
    with wire_server(member.reply) as wire, gateway.scenario() as scenario:
        model: Final = member.deployment(scenario, wire.url, None)
        box: Final = (
            _decide(gateway, model, route=route, model_info=value)
            if route == "/v1/decisions"
            else _decide_in_system_one_format(gateway, model, route, model_info=value)
        )
        _assert_refused_box(box)
        assert wire.drain() == ()
        predicate_only: Final = _decide(gateway, model, (_PREDICATE,), model_info=value)
        assert predicate_only.status_code == 200, predicate_only.text
        (call,) = wire.drain()
        assert _JSON_OBJECT.validate_json(call.body) == member.upstream_body((_PREDICATE,))


def test_a_duplicated_or_metadata_model_info_and_a_forged_id_change_nothing_behind_the_router(
    gateway: Gateway,
) -> None:
    member: Final = _perplexity()
    forged: Final = f"forged-{uuid.uuid4().hex}"
    with wire_server(member.reply) as wire, gateway.scenario() as scenario:
        model: Final = member.deployment(scenario, wire.url, None)
        first: Final = json.dumps(
            {
                "model": model,
                "input": _INPUT,
                "questions": list(_QUESTIONS),
                "num_retries": 0,
                "model_info": {"supports_box_questions": False},
            }
        )
        duplicated: Final = (
            f'{first[:-1]}, "model_info": {json.dumps({"supports_box_questions": True, "id": forged})}}}'
        )
        raw: Final = gateway.client.post(
            "/v1/decisions",
            content=duplicated.encode(),
            headers={"Authorization": f"Bearer {gateway.key}", "Content-Type": "application/json"},
        )
        _assert_refused_box(raw)
        _assert_refused_box(_decide(gateway, model, metadata={"model_info": {"supports_box_questions": True}}))
        assert wire.drain() == ()
        answered: Final = _decide(gateway, model, (_PREDICATE,), model_info={"id": forged})
        assert answered.status_code == 200, answered.text
        assert answered.headers.get("x-litellm-model-id", "") not in ("", forged), answered.headers
        assert _spend_row(answered.headers["x-litellm-call-id"])["model_id"] == answered.headers["x-litellm-model-id"]
        (call,) = wire.drain()
        assert _JSON_OBJECT.validate_json(call.body) == member.upstream_body((_PREDICATE,))


@pytest.mark.parametrize("status", (401, 404, 413))
def test_a_flagged_box_request_brings_the_provider_error_back_after_one_attempt(gateway: Gateway, status: int) -> None:
    member: Final = _perplexity()
    marker: Final = f"provider says {status} {uuid.uuid4().hex}"

    def failing(_request: Request) -> Reply:
        return Reply(status=status, body=json.dumps({"error": {"message": marker}}).encode())

    with wire_server(failing) as wire, gateway.scenario() as scenario:
        model: Final = member.deployment(scenario, wire.url, {"supports_box_questions": True})
        response: Final = _decide(gateway, model)
        assert response.status_code == status, response.text
        assert marker in response.text, response.text
        (call,) = wire.drain()
        assert _JSON_OBJECT.validate_json(call.body) == member.upstream_body(_QUESTIONS)


def test_a_box_question_answered_with_the_wrong_answer_type_reaches_the_caller_as_a_refusal(gateway: Gateway) -> None:
    member: Final = _perplexity()

    def mismatched(_request: Request) -> Reply:
        answers: Final = {"cracked": {"type": "noul", "noul": 0.9}, "crack": {"type": "noul", "noul": 0.4}}
        return Reply(body=json.dumps({"model": member.body_model, "answers": answers, "usage": _USAGE}).encode())

    with wire_server(mismatched) as wire, gateway.scenario() as scenario:
        model: Final = member.deployment(scenario, wire.url, {"supports_box_questions": True})
        response: Final = _decide(gateway, model)
        assert response.status_code == 200, response.text
        assert response.json()["answers"] == [_PREDICATE_ANSWER, _BOX_REFUSAL], response.text
        assert len(wire.drain()) == 1


@pytest.mark.parametrize(
    ("header", "answers"), ((None, False), ("false", False), ("true", True)), ids=("absent", "false", "true")
)
def test_typesafe_needs_its_images_header_for_boxes_even_on_a_flagged_deployment(
    gateway: Gateway, header: str | None, answers: bool
) -> None:
    template: Final = _members(_tag())["typesafe"]
    member: Final = _Member(
        template.provider,
        template.model,
        template.path,
        template.body_model,
        headers=MappingProxyType({} if header is None else {_IMAGES_HEADER: header}),
    )
    with wire_server(member.reply) as wire, gateway.scenario() as scenario:
        model: Final = member.deployment(scenario, wire.url, {"supports_box_questions": True})
        response: Final = _decide(gateway, model)
        if not answers:
            _assert_refused_box(response)
            assert wire.drain() == ()
            return
        assert response.status_code == 200, response.text
        assert response.json()["answers"] == _expected_answers(_QUESTIONS)
        (call,) = wire.drain()
        assert call.headers.get(_IMAGES_HEADER.lower()) == "true"
        assert _JSON_OBJECT.validate_json(call.body) == member.upstream_body(_QUESTIONS)


@pytest.mark.parametrize(
    ("flag", "keeps_box"), ((True, True), (False, False), (None, False)), ids=("true", "false", "unset")
)
def test_drop_params_drops_box_questions_unless_the_deployment_flag_is_true(
    gateway: Gateway, flag: bool | None, keeps_box: bool
) -> None:
    member: Final = _perplexity()
    with wire_server(member.reply) as wire, gateway.scenario() as scenario:
        model_info: Final[dict[str, JsonValue] | None] = None if flag is None else {"supports_box_questions": flag}
        model: Final = member.deployment(scenario, wire.url, model_info, drop_params=True)
        response: Final = _decide(gateway, model)
        assert response.status_code == 200, response.text
        sent: Final = _QUESTIONS if keeps_box else (_PREDICATE,)
        answers: Final = _expected_answers(_QUESTIONS) if keeps_box else [_PREDICATE_ANSWER, _BOX_REFUSAL]
        assert response.json()["answers"] == answers, response.text
        (call,) = wire.drain()
        assert _JSON_OBJECT.validate_json(call.body) == member.upstream_body(sent)
        if keeps_box:
            return
        nothing_left: Final = _decide(gateway, model, (_BOX_QUESTION,))
        assert nothing_left.status_code == 400, nothing_left.text
        assert "leaves no questions to answer" in nothing_left.text, nothing_left.text
        assert wire.drain() == ()


@pytest.mark.timeout(2 * graceful_stop_seconds() + 120)
def test_a_proxy_without_a_router_ignores_a_body_model_info_for_box_questions_and_model_ids(
    gateway: Gateway, tmp_path: Path
) -> None:
    member: Final = _perplexity()
    forged: Final = f"forged-{uuid.uuid4().hex}"
    body_model_info: Final[dict[str, JsonValue]] = {"supports_box_questions": True, "id": forged}
    with wire_server(member.reply) as wire:
        with configless_proxy_process(
            gateway, tmp_path, {"PERPLEXITYAI_API_KEY": _API_KEY}, model=member.model, api_base=wire.url
        ) as owned:
            box: Final = _decide(owned.gateway, member.model, model_info=body_model_info)
            predicate_only: Final = _decide(owned.gateway, member.model, (_PREDICATE,), model_info=body_model_info)
            system_one: Final = owned.gateway.request(
                "POST",
                "/v1/systemone",
                {
                    "model": member.model,
                    "state": _INPUT,
                    "questions": {"cracked": _systemone_question(_PREDICATE)},
                    "model_info": body_model_info,
                },
            )
            _assert_refused_box(box)
            for answered in (predicate_only, system_one):
                assert answered.status_code == 200, answered.text
                assert answered.headers.get("x-litellm-model-id", "") != forged, answered.headers
                assert _spend_row(answered.headers["x-litellm-call-id"])["model_id"] != forged
        assert [_JSON_OBJECT.validate_json(call.body) for call in wire.drain()] == [
            member.upstream_body((_PREDICATE,))
        ] * 2


def _new_deployment(gateway: Gateway, scenario: Scenario, member: _Member, wire: Wire) -> tuple[str, str]:
    name: Final = f"integration-{uuid.uuid4().hex}"
    created: Final = gateway.post(
        "/model/new",
        {
            "model_name": name,
            "litellm_params": {"model": member.model, "api_base": wire.url, "api_key": _API_KEY},
            "model_info": {"supports_box_questions": True},
        },
    )
    identity: Final = string_value(object_value(created["model_info"])["id"])
    scenario.cleanups.callback(scenario.delete_model, identity)
    return name, identity


def _box_outcome(gateway: Gateway, model: str, outcomes: SimpleQueue[int]) -> int:
    response: Final = _decide(gateway, model, (_BOX_QUESTION,))
    outcomes.put(response.status_code)
    assert response.status_code in (200, 400), response.text
    if response.status_code == 400:
        assert _REFUSED_BOX in response.text, response.text
        return 400
    assert response.json()["answers"] == [_BOX_ANSWER], response.text
    return 200


@pytest.mark.timeout(240)
def test_turning_the_flag_off_under_traffic_converges_without_sending_boxes_to_a_refusing_deployment(
    gateway: Gateway,
) -> None:
    member: Final = _perplexity()
    outcomes: Final[SimpleQueue[int]] = SimpleQueue()
    with wire_server(member.reply) as wire, gateway.scenario() as scenario:
        model, identity = _new_deployment(gateway, scenario, member, wire)
        before: Final = tuple(_box_outcome(gateway, model, outcomes) for _ in range(6))
        assert before == (200,) * 6, before
        patched: Final = gateway.request(
            "PATCH", f"/model/{identity}/update", {"model_info": {"id": identity, "supports_box_questions": False}}
        )
        assert patched.status_code == 200, patched.text
        during: Final = eventually(
            lambda: tuple(_box_outcome(gateway, model, outcomes) for _ in range(8)),
            lambda outcomes: outcomes == (400,) * 8,
            seconds=3 * PROXY_CONFIG_RELOAD_INTERVAL_SECONDS,
        )
        assert during == (400,) * 8
        received: Final = wire.drain()
        statuses: Final = tuple(outcomes.get_nowait() for _ in range(outcomes.qsize()))
        assert all(_JSON_OBJECT.validate_json(call.body) == member.upstream_body((_BOX_QUESTION,)) for call in received)
        assert len(received) == statuses.count(200), statuses
