import json

import httpx
import pytest
from openai import AsyncOpenAI

from litellm.models.task_insights import TASK_INSIGHTS_ANALYSIS_TAG
from litellm.task_insights.gateway import ANALYSIS_HEADERS, EMBED_BATCH_SIZE, GatewayEmbedder, GatewaySynthesizer


def _client(handler) -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url="http://gateway/v1",
        api_key="sk-test",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        default_headers=ANALYSIS_HEADERS,
        max_retries=0,
    )


def _completion(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "c",
            "object": "chat.completion",
            "created": 0,
            "model": "claude",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        },
    )


@pytest.mark.asyncio
async def test_embeddings_are_batched_returned_in_input_order_and_tagged_as_analysis():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        inputs = json.loads(request.content)["input"]
        data = [
            {"object": "embedding", "index": index, "embedding": [float(len(text))]}
            for index, text in enumerate(inputs)
        ]
        return httpx.Response(200, json={"object": "list", "data": list(reversed(data)), "model": "qwen"})

    texts = tuple("x" * length for length in range(1, 45))

    vectors = await GatewayEmbedder(_client(handler), "qwen").embed(texts)

    assert vectors == tuple((float(len(text)),) for text in texts)
    sizes = [len(json.loads(request.content)["input"]) for request in seen]
    assert sum(sizes) == len(texts)
    assert max(sizes) <= EMBED_BATCH_SIZE <= 20
    assert len(seen) == -(-len(texts) // EMBED_BATCH_SIZE)
    assert {request.headers["x-litellm-tags"] for request in seen} == {TASK_INSIGHTS_ANALYSIS_TAG}


@pytest.mark.asyncio
async def test_distillation_keeps_one_intent_per_request_and_falls_back_to_excerpts_on_a_bad_reply():
    replies = iter(['Here you go: ["fix a test", "add an endpoint"]', "not json"])
    synthesizer = GatewaySynthesizer(_client(lambda request: _completion(next(replies))), "claude")

    assert await synthesizer.distill(("first", "second")) == ("fix a test", "add an endpoint")
    assert await synthesizer.distill(("only text",)) == ("only text",)


@pytest.mark.asyncio
async def test_distillation_rejects_a_reply_with_the_wrong_number_of_intents():
    synthesizer = GatewaySynthesizer(_client(lambda request: _completion('["one"]')), "claude")

    assert await synthesizer.distill(("a", "b")) == ("a", "b")


@pytest.mark.asyncio
async def test_synthesis_parses_the_definition_and_names_an_unusable_reply_for_review():
    good = {
        "name": "Debug failing tests",
        "description": "Find why tests fail",
        "inclusion_criteria": "A failing test",
        "exclusion_criteria": "New tests",
        "required_capabilities": ["function_calling"],
        "example_indices": [0],
    }
    replies = iter([f"```json\n{json.dumps(good)}\n```", '{"name": ""}'])
    prompts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        prompts.append(json.loads(request.content)["messages"][0]["content"])
        return _completion(next(replies))

    synthesizer = GatewaySynthesizer(_client(handler), "claude")

    parsed = await synthesizer.synthesize(("the build fails",))
    fallback = await synthesizer.synthesize(("the build fails",))

    assert (parsed.name, parsed.required_capabilities) == ("Debug failing tests", ("function_calling",))
    assert fallback.name == "Unnamed group"
    assert '0. "the build fails"' in prompts[0]


@pytest.mark.asyncio
async def test_criteria_returned_as_bullet_lists_still_make_a_named_task():
    reply = {
        "name": "Debug failing builds and tests",
        "description": "Find and fix why a build or test run fails.",
        "inclusion_criteria": [
            "Reports a failing test or build",
            "Asks for the cause or a fix.",
            " ",
        ],
        "exclusion_criteria": ["Writing new tests", "Adding features"],
        "required_capabilities": ["function_calling", "reasoning"],
        "example_indices": [0, 2],
    }
    synthesizer = GatewaySynthesizer(_client(lambda request: _completion(json.dumps(reply))), "claude")

    task = await synthesizer.synthesize(("a", "b", "c"))

    assert task.name == "Debug failing builds and tests"
    assert task.inclusion_criteria == "Reports a failing test or build. Asks for the cause or a fix."
    assert task.exclusion_criteria == "Writing new tests. Adding features."
    assert (task.required_capabilities, task.example_indices) == (("function_calling", "reasoning"), (0, 2))


@pytest.mark.asyncio
async def test_an_unsupported_capability_is_not_silently_dropped():
    reply = {"name": "Read screenshots", "description": "d", "required_capabilities": ["telepathy"]}
    synthesizer = GatewaySynthesizer(_client(lambda request: _completion(json.dumps(reply))), "claude")

    task = await synthesizer.synthesize(("a",))

    assert task.name == "Unnamed group"
    assert task.required_capabilities == ()
