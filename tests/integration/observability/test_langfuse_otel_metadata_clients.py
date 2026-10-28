import json
import uuid
from pathlib import Path
from typing import Final

import httpx
from _langfuse_otel_metadata_support import (
    assert_identity,
    chat_identity_fields,
    exported_llm_span,
    langfuse_rig,
    scenario_identity,
)
from anthropic import Anthropic, AsyncAnthropic
from integration._support.client import Gateway
from openai import AsyncOpenAI, OpenAI


def test_langfuse_otel_metadata_reaches_langfuse_for_openai_sdk_stream(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "lf-sdkstream-" + uuid.uuid4().hex
    with (
        langfuse_rig(gateway, tmp_path, {}, marker) as rig,
        rig.candidate.scenario() as scenario,
    ):
        key, identity = scenario_identity(scenario, rig.provider)
        with OpenAI(
            base_url=str(rig.candidate.client.base_url).rstrip("/") + "/v1",
            api_key=key,
            http_client=httpx.Client(trust_env=False, timeout=30),
        ) as client:
            chunks: Final = tuple(
                client.chat.completions.create(
                    model=identity["model"],
                    messages=[{"role": "user", "content": marker}],
                    stream=True,
                    user=identity["end_user"],
                    extra_body={"cache": {"no-cache": True}},
                )
            )
        assert chunks, "stream produced no chunks"
        response_id: Final = chunks[-1].id
        attrs: Final = exported_llm_span(rig.collector, response_id)
        assert_identity(attrs, chat_identity_fields(identity))


async def test_langfuse_otel_metadata_reaches_langfuse_for_openai_sdk_async(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "lf-sdkasync-" + uuid.uuid4().hex
    with (
        langfuse_rig(gateway, tmp_path, {}, marker) as rig,
        rig.candidate.scenario() as scenario,
    ):
        key, identity = scenario_identity(scenario, rig.provider)
        async with AsyncOpenAI(
            base_url=str(rig.candidate.client.base_url).rstrip("/") + "/v1",
            api_key=key,
            http_client=httpx.AsyncClient(trust_env=False, timeout=30),
        ) as client:
            response: Final = await client.chat.completions.create(
                model=identity["model"],
                messages=[{"role": "user", "content": marker}],
                user=identity["end_user"],
                extra_body={"cache": {"no-cache": True}},
            )
        attrs: Final = exported_llm_span(rig.collector, response.id)
        assert_identity(attrs, chat_identity_fields(identity))


def test_langfuse_otel_metadata_reaches_langfuse_for_anthropic_sdk_messages(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "lf-anthropic-" + uuid.uuid4().hex
    with (
        langfuse_rig(gateway, tmp_path, {}, marker) as rig,
        rig.candidate.scenario() as scenario,
    ):
        key, identity = scenario_identity(scenario, rig.provider)
        model: Final = scenario.model(model="anthropic/claude-sonnet-4-6", api_base=rig.provider.url)
        with Anthropic(
            base_url=str(rig.candidate.client.base_url).rstrip("/"),
            api_key=key,
            http_client=httpx.Client(trust_env=False, timeout=30),
        ) as client:
            message: Final = client.messages.create(
                model=model,
                max_tokens=64,
                messages=[{"role": "user", "content": marker}],
                extra_body={"litellm_metadata": {"tags": ["lf8283"]}, "metadata": {"user_id": identity["end_user"]}},
            )
        attrs: Final = exported_llm_span(rig.collector, message.id)
        assert_identity(attrs, chat_identity_fields(identity))


async def test_langfuse_otel_metadata_reaches_langfuse_for_anthropic_sdk_stream(
    gateway: Gateway, tmp_path: Path
) -> None:
    marker: Final = "lf-anthstream-" + uuid.uuid4().hex
    with (
        langfuse_rig(gateway, tmp_path, {}, marker) as rig,
        rig.candidate.scenario() as scenario,
    ):
        key, identity = scenario_identity(scenario, rig.provider)
        model: Final = scenario.model(model="anthropic/claude-sonnet-4-6", api_base=rig.provider.url)
        async with (
            AsyncAnthropic(
                base_url=str(rig.candidate.client.base_url).rstrip("/"),
                api_key=key,
                http_client=httpx.AsyncClient(trust_env=False, timeout=30),
            ) as client,
            client.messages.stream(
                model=model,
                max_tokens=64,
                messages=[{"role": "user", "content": marker}],
                extra_body={"metadata": {"user_id": identity["end_user"]}},
            ) as stream,
        ):
            final: Final = await stream.get_final_message()
        attrs: Final = exported_llm_span(rig.collector, final.id)
        assert_identity(attrs, chat_identity_fields(identity))


def test_langfuse_otel_metadata_reaches_langfuse_for_responses_api(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "lf-responses-" + uuid.uuid4().hex
    with (
        langfuse_rig(gateway, tmp_path, {}, marker) as rig,
        rig.candidate.scenario() as scenario,
    ):
        key, identity = scenario_identity(scenario, rig.provider)
        client: Final = OpenAI(
            base_url=str(rig.candidate.client.base_url).rstrip("/") + "/v1",
            api_key=key,
            http_client=httpx.Client(trust_env=False, timeout=30),
        )
        response: Final = client.responses.create(
            model=identity["model"],
            input=marker,
            user=identity["end_user"],
            extra_body={"cache": {"no-cache": True}},
        )
        attrs: Final = exported_llm_span(rig.collector, response.id)
        assert_identity(attrs, chat_identity_fields(identity))


def test_langfuse_otel_metadata_reaches_langfuse_for_responses_sse_stream(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "lf-respstream-" + uuid.uuid4().hex
    with (
        langfuse_rig(gateway, tmp_path, {}, marker) as rig,
        rig.candidate.scenario() as scenario,
    ):
        key, identity = scenario_identity(scenario, rig.provider)
        with rig.candidate.client.stream(
            "POST",
            "/v1/responses",
            json={
                "model": identity["model"],
                "input": marker,
                "user": identity["end_user"],
                "stream": True,
                "cache": {"no-cache": True},
            },
            headers={"Authorization": f"Bearer {key}"},
        ) as stream:
            events: Final = tuple(
                json.loads(line.removeprefix("data: "))
                for line in stream.iter_lines()
                if line.startswith("data: ") and line != "data: [DONE]"
            )
        response_id: Final = next(
            event["response_id"] for event in events if event.get("type") == "response.output_text.delta"
        )
        attrs: Final = exported_llm_span(rig.collector, response_id)
        assert_identity(attrs, chat_identity_fields(identity))
