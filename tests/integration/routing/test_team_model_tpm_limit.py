import json
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, eventually
from integration._support.wire import Reply, Request, wire_server

PROVIDER_MODEL: Final = "gpt-4o-mini"
TEAM_MODEL_TPM: Final = 100
MAX_TOKENS: Final = 60
CONCURRENT_REQUESTS: Final = 3
UPSTREAM_REPLY: Final = json.dumps(
    {
        "id": "chatcmpl-team-tpm-control",
        "object": "chat.completion",
        "created": 1,
        "model": PROVIDER_MODEL,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "team tpm control"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
    }
).encode()


@pytest.mark.covers("routing.team_model_tpm.concurrent_requests_over_the_limit_are_rejected_before_the_provider_call")
def test_concurrent_team_model_tpm_requests_reserve_tokens_before_reaching_the_provider(gateway: Gateway) -> None:
    probe: Final = "team tpm probe " + uuid.uuid4().hex
    release: Final = threading.Event()

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer synthetic-team-tpm-key"
        body: Final = json.loads(request.body)
        content: Final = body["messages"][0]["content"]
        assert body == {
            "model": PROVIDER_MODEL,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": MAX_TOKENS,
        }
        assert content.startswith(probe), content
        release.wait(timeout=10)
        return Reply(body=UPSTREAM_REPLY)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"openai/{PROVIDER_MODEL}",
            api_base=wire.url + "/v1",
            api_key="synthetic-team-tpm-key",
        )
        team: Final = scenario.team(metadata={"model_tpm_limit": {model: TEAM_MODEL_TPM}})
        key: Final = scenario.key(team_id=team)

        def send(index: int) -> httpx.Response:
            return gateway.request(
                "POST",
                "/v1/chat/completions",
                {
                    "model": model,
                    "max_tokens": MAX_TOKENS,
                    "messages": [{"role": "user", "content": f"{probe} {index}"}],
                },
                key=key,
            )

        with ThreadPoolExecutor(max_workers=CONCURRENT_REQUESTS) as pool:
            futures: Final[tuple[Future[httpx.Response], ...]] = tuple(
                pool.submit(send, index) for index in range(CONCURRENT_REQUESTS)
            )
            eventually(
                lambda: sum(future.done() for future in futures) + wire.received.qsize(),
                lambda settled: settled >= CONCURRENT_REQUESTS,
                seconds=10,
            )
            release.set()
            responses: Final = tuple(future.result(timeout=15) for future in futures)
        statuses: Final = tuple(sorted(response.status_code for response in responses))
        assert statuses == (200, 429, 429), tuple(response.text for response in responses)
        assert len(wire.drain()) == 1, statuses
        served: Final = next(response for response in responses if response.status_code == 200)
        assert served.json()["usage"] == {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}, served.text
        for rejected in (response for response in responses if response.status_code == 429):
            error: Final = rejected.json()["error"]
            assert (error["type"], error["code"], error["param"]) == ("throttling_error", "429", None), rejected.text
            assert f"Limit type: tokens. Current limit: {TEAM_MODEL_TPM}," in error["message"], rejected.text
