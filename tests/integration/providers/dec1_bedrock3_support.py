"""Shared rig for the third round of December 1 AWS Bedrock contracts (`test_dec1_bedrock3_*`).

Round 3 covers the Bedrock surfaces the first two rounds left open: ARN model ids (application
inference profiles, prompt routers, provisioned throughput, imported models), the Invoke families that
render their own prompt (Llama 3 and 4, DeepSeek R1, Qwen 3) or speak OpenAI (imported OpenAI models,
gpt-oss), Converse document, video and citation blocks, Bedrock Agents (`bedrock/agent/...`) and
AgentCore runtimes (`bedrock/agentcore/...`), knowledge base retrieval through `/v1/vector_stores`,
rerank, async-invoke and batch embeddings, the 403 AccessDenied shape real Bedrock sends, and the
Bedrock Mantle `/v1/messages` route.

Every helper here points the proxy at an owned TCP peer (`wire_server`) so no test can reach a real
AWS host: deployments always carry `api_base` or `aws_bedrock_runtime_endpoint` (or, for batch files,
`s3_endpoint_url` and `aws_bedrock_control_plane_endpoint`) set to the peer. The generic Converse and
Invoke helpers are reused from `dec1_bedrock_support.py` and `dec1_msgprov_support.py`, which stay
unchanged.
"""

from __future__ import annotations

import base64
import json
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Final
from urllib.parse import quote

import httpx
from integration._support.client import Gateway, Scenario
from integration._support.upstream import _aws_event_frame
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.messages_endpoint.dec1_msgprov_support import EVENT_STREAM
from integration.providers.dec1_bedrock_support import TOKEN, Rig
from pydantic import JsonValue

ACCOUNT: Final = "123456789012"
"""A synthetic AWS account id used inside every ARN."""

APP_PROFILE_ARN: Final = f"arn:aws:bedrock:us-west-2:{ACCOUNT}:application-inference-profile/dec1prof3abc"
"""An application inference profile ARN: its tail is an opaque id that names no model or provider."""
PROMPT_ROUTER_ARN: Final = f"arn:aws:bedrock:us-east-1:{ACCOUNT}:default-prompt-router/anthropic.claude:1"
"""A Bedrock default prompt router ARN. Its tail names the `anthropic` family."""
PROVISIONED_ARN: Final = f"arn:aws:bedrock:us-east-1:{ACCOUNT}:provisioned-model/dec1pt9xyz"
"""A provisioned throughput model ARN: opaque tail, no provider in the id."""
IMPORTED_ARN: Final = f"arn:aws:bedrock:us-east-1:{ACCOUNT}:imported-model/dec1imp7k2m"
"""A custom model import ARN; the family comes from a `llama/`, `deepseek_r1/`, `qwen3/` or `openai/` prefix."""
AGENTCORE_ARN: Final = f"arn:aws:bedrock-agentcore:us-west-2:{ACCOUNT}:runtime/dec1_agent-AbCdEf1234"

AGENT_ID: Final = "DEC1AGENT1"
AGENT_ALIAS_ID: Final = "DEC1ALIAS1"

SONNET_45: Final = "anthropic.claude-sonnet-4-5-20250929-v1:0"
SONNET_45_INPUT: Final = 3e-06
"""`input_cost_per_token` of the `anthropic.claude-sonnet-4-5-20250929-v1:0` row in the cost map."""
SONNET_45_OUTPUT: Final = 1.5e-05
"""`output_cost_per_token` of the `anthropic.claude-sonnet-4-5-20250929-v1:0` row in the cost map."""

LLAMA3_8B: Final = "meta.llama3-8b-instruct-v1:0"
LLAMA4_MAVERICK: Final = "meta.llama4-maverick-17b-instruct-v1:0"
GPT_OSS_120B: Final = "openai.gpt-oss-120b-1:0"
QWEN3_32B: Final = "qwen.qwen3-32b-v1:0"
DEEPSEEK_V3: Final = "deepseek.v3-v1:0"

LLAMA3_BOS: Final = "<|begin_of_text|>"
"""`bos_token` of the baked-in `meta-llama/Meta-Llama-3-8B-Instruct` config in `litellm/constants.py`."""
DEEPSEEK_BOS: Final = "<｜begin▁of▁sentence｜>"
"""`bos_token.content` of the baked-in `deepseek-r1/deepseek-r1-7b-instruct` config (full-width bars, U+2581)."""
DEEPSEEK_USER: Final = "<｜User｜>"
DEEPSEEK_ASSISTANT: Final = "<｜Assistant｜>"

KB_SECRET: Final = "synthetic-knowledge-base-secret-key-dec1"
KB_ACCESS_KEY: Final = "AKIADEC1BEDROCK3KB01"


def quoted(model_id: str) -> str:
    """A Bedrock model id or ARN as one URL path segment (`urllib.parse.quote(..., safe="")`)."""
    return quote(model_id, safe="")


def unique_text(prefix: str = "dec1-bedrock3") -> str:
    """A fresh prompt, so the shared proxy's response cache never answers a test."""
    return f"{prefix}-{uuid.uuid4().hex}"


def response_cost(response: httpx.Response) -> float:
    """The `x-litellm-response-cost` header the proxy puts on every priced answer."""
    return float(response.headers["x-litellm-response-cost"])


def json_body(request: Request) -> dict[str, JsonValue]:
    decoded: Final = json.loads(request.body)
    assert isinstance(decoded, dict), request.body
    return decoded


def json_reply(payload: Mapping[str, JsonValue] | list[JsonValue], *, status: int = 200) -> Reply:
    return Reply(status=status, body=json.dumps(payload).encode())


def only_posts(wire: Wire) -> tuple[Request, ...]:
    """Every POST the peer received; GET health probes are dropped."""
    return tuple(request for request in wire.drain() if request.method == "POST")


# ------------------------------------------------------------------ Bedrock Agents (InvokeAgent)


def agent_chunk(text: str) -> bytes:
    """One InvokeAgent `chunk` event: `{"bytes": <base64 text>}` in an AWS event stream frame."""
    return _aws_event_frame("chunk", {"bytes": base64.b64encode(text.encode()).decode()}, "sc", "u")


def agent_trace(trace: Mapping[str, JsonValue]) -> bytes:
    """One InvokeAgent `trace` event whose JSON payload is `{"trace": <trace>}`."""
    return _aws_event_frame("trace", {"trace": dict(trace)}, "sc", "u")


def agent_reply(*frames: bytes) -> Reply:
    return Reply(content_type=EVENT_STREAM, chunks=frames)


def preprocessing_usage(input_tokens: int, output_tokens: int) -> dict[str, JsonValue]:
    """A `preProcessingTrace` carrying model usage, the only place InvokeAgent reports tokens."""
    return {
        "preProcessingTrace": {
            "modelInvocationOutput": {
                "metadata": {"usage": {"inputTokens": input_tokens, "outputTokens": output_tokens}}
            }
        }
    }


# ------------------------------------------------------------------ Knowledge bases


@dataclass(frozen=True, slots=True)
class KnowledgeBase:
    gateway: Gateway
    wire: Wire
    identity: str

    def search(self, **body: JsonValue) -> httpx.Response:
        return self.gateway.request("POST", f"/v1/vector_stores/{self.identity}/search", dict(body))

    def one_post(self) -> Request:
        posts: Final = only_posts(self.wire)
        assert len(posts) == 1, [(request.method, request.target) for request in posts]
        return posts[0]


@contextmanager
def knowledge_base(
    gateway: Gateway, respond: Callable[[Request], Reply], **litellm_params: JsonValue
) -> Iterator[KnowledgeBase]:
    """Register a Bedrock knowledge base (`custom_llm_provider: bedrock`) whose agent runtime is the peer.

    The vector store is created through `POST /vector_store/new` with SigV4 static keys and
    `aws_bedrock_runtime_endpoint` at the peer (the same shape as
    `test_bedrock_knowledge_base_user_context_wire.py`), and deleted when the block ends.
    """
    identity: Final = f"KB{uuid.uuid4().hex[:8].upper()}"
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        gateway.post(
            "/vector_store/new",
            {
                "vector_store_id": identity,
                "custom_llm_provider": "bedrock",
                "litellm_params": {
                    "aws_region_name": "us-east-1",
                    "aws_access_key_id": KB_ACCESS_KEY,
                    "aws_secret_access_key": KB_SECRET,
                    "aws_bedrock_runtime_endpoint": wire.url,
                    **litellm_params,
                },
            },
        )
        scenario.cleanups.callback(gateway.post, "/vector_store/delete", {"vector_store_id": identity})
        yield KnowledgeBase(gateway, wire, identity)


def retrieve_reply(*results: Mapping[str, JsonValue]) -> Reply:
    return json_reply({"retrievalResults": [dict(result) for result in results]})


def text_hit(text: str, score: float, uri: str) -> dict[str, JsonValue]:
    return {
        "content": {"text": text, "type": "TEXT"},
        "score": score,
        "metadata": {"x-amz-bedrock-kb-source-uri": uri, "x-amz-bedrock-kb-chunk-id": "chunk-1"},
    }


# ------------------------------------------------------------------ Model registration


def deployment(
    scenario: Scenario, *, model: str, model_info: Mapping[str, JsonValue] | None = None, **params: JsonValue
) -> str:
    """`POST /model/new` with exactly `params` (no harness `api_key`/`api_base` defaults); deleted on exit."""
    name: Final = f"dec1-bedrock3-{uuid.uuid4().hex}"
    created: Final = scenario.gateway.post(
        "/model/new",
        {
            "model_name": name,
            "litellm_params": {"model": model, **{key: value for key, value in params.items() if value is not None}},
            "model_info": dict(model_info) if model_info is not None else {},
        },
    )
    identity: Final = created["model_info"]["id"]  # pyright: ignore[reportIndexIssue, reportArgumentType, reportCallIssue, reportOptionalSubscript]
    scenario.cleanups.callback(scenario.delete_model, str(identity))
    return name


# ------------------------------------------------------------------ AgentCore runtimes


@contextmanager
def agentcore(gateway: Gateway, respond: Callable[[Request], Reply], **params: JsonValue) -> Iterator[Rig]:
    """A `bedrock/agentcore/<AGENTCORE_ARN>` deployment whose runtime host is an owned peer.

    AgentCore ignores `api_base` and reads only `aws_bedrock_runtime_endpoint` for its host
    (`litellm/llms/bedrock/chat/agentcore/transformation.py:104`), so the peer URL goes there; without it the
    proxy would dial `bedrock-agentcore.us-west-2.amazonaws.com`. Defaults: Bearer `api_key` and
    `aws_region_name: us-west-2`; a `None` value drops a parameter (pass `sigv4_params()` for SigV4).
    """
    with wire_server(_posts_only(respond)) as wire, gateway.scenario() as scenario:
        merged: Final[dict[str, JsonValue]] = {
            "api_key": TOKEN,
            "aws_region_name": "us-west-2",
            "aws_bedrock_runtime_endpoint": wire.url,
            **params,
        }
        name: Final = deployment(scenario, model=f"bedrock/agentcore/{AGENTCORE_ARN}", **merged)
        yield Rig(gateway, scenario, wire, name)


def _posts_only(respond: Callable[[Request], Reply]) -> Callable[[Request], Reply]:
    def answer(request: Request) -> Reply:
        return respond(request) if request.method == "POST" else Reply(body=b"{}")

    return answer
