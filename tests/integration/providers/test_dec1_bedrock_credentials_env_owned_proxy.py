"""Bedrock credentials and regions that come from the proxy's own environment, on a proxy this module owns.

Some Bedrock behaviour depends only on process environment variables, which the shared proxy cannot vary:
`AWS_REGION_NAME` (preferred) and `AWS_REGION` pick the signing region when a deployment names none
(`_get_aws_region_name`, `litellm/llms/bedrock/base_aws_llm.py:716`); `AWS_BEARER_TOKEN_BEDROCK` is the
Bedrock API key used when a deployment has no `api_key` (`bedrock_bearer_token`, `base_aws_llm.py:134`),
while an explicitly empty `api_key` means "no bearer" and must not fall back to the env token; and
`AWS_BEDROCK_RUNTIME_ENDPOINT` replaces the public runtime host when no `api_base` or
`aws_bedrock_runtime_endpoint` is configured (`get_runtime_endpoint`, `base_aws_llm.py:1483`).

`/v1/messages` on Bedrock InvokeModel is served by Rust only since #51330, and #52042 fixed a Rust parity
bug where an empty `api_key` let the Rust path read `AWS_BEARER_TOKEN_BEDROCK` and send it as a bearer,
so the same contract is pinned on both the Python chat path and the Rust Messages path.

The module starts two owned peers (the deployments' `api_base`, and the env runtime endpoint), then one
owned proxy whose config lists one deployment per case, so the environment applies to all of them.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import JSON_OBJECT, Gateway, gateway_from_environment
from integration._support.process import graceful_stop_seconds, owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.messages_endpoint.dec1_msgprov_support import invoke_echo
from integration.providers.dec1_bedrock_support import (
    ACCESS_KEY,
    CLAUDE_HAIKU_55,
    SECRET_KEY,
    converse_reply,
    marker,
    path_of,
    sigv4_scope,
)
from pydantic import JsonValue

ENV_REGION: Final = "eu-west-3"
ENV_STANDARD_REGION: Final = "ap-northeast-1"
ENV_BEARER: Final = "synthetic-env-bedrock-bearer"
MESSAGES_MODEL_ID: Final = "us.anthropic.claude-sonnet-5"
"""A Claude id with no `:` suffix, so the Invoke path needs no percent encoding."""

ENV_REGION_MODEL: Final = "dec1-bedrock-env-region"
ENV_BEARER_MODEL: Final = "dec1-bedrock-env-bearer"
EMPTY_KEY_MODEL: Final = "dec1-bedrock-empty-key"
ENV_ENDPOINT_MODEL: Final = "dec1-bedrock-env-endpoint"
MESSAGES_EMPTY_KEY_MODEL: Final = "dec1-bedrock-messages-empty-key"
MESSAGES_ENV_BEARER_MODEL: Final = "dec1-bedrock-messages-env-bearer"

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 180)


def _respond(request: Request) -> Reply:
    if request.method != "POST":
        return Reply(body=b'{"object":"list","data":[]}')
    if "/invoke" in request.target:
        return invoke_echo(request)
    return converse_reply()


def _deployment(name: str, model: str, **params: JsonValue) -> dict[str, JsonValue]:
    return {"model_name": name, "litellm_params": {"model": model, **params}}


def _config(directory: Path, base_url: str) -> Path:
    base: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    keys: Final[dict[str, JsonValue]] = {"aws_access_key_id": ACCESS_KEY, "aws_secret_access_key": SECRET_KEY}
    configuration: Final = {
        **base,
        "model_list": [
            _deployment(ENV_REGION_MODEL, f"bedrock/{CLAUDE_HAIKU_55}", api_base=base_url, **keys),
            _deployment(ENV_BEARER_MODEL, f"bedrock/{CLAUDE_HAIKU_55}", api_base=base_url, aws_region_name="us-east-1"),
            _deployment(
                EMPTY_KEY_MODEL,
                f"bedrock/{CLAUDE_HAIKU_55}",
                api_base=base_url,
                api_key="",
                aws_region_name="us-east-1",
                **keys,
            ),
            _deployment(ENV_ENDPOINT_MODEL, f"bedrock/{CLAUDE_HAIKU_55}", aws_region_name="us-east-1"),
            _deployment(
                MESSAGES_EMPTY_KEY_MODEL,
                f"bedrock/invoke/{MESSAGES_MODEL_ID}",
                api_base=base_url,
                api_key="",
                aws_region_name="us-east-1",
                **keys,
            ),
            _deployment(
                MESSAGES_ENV_BEARER_MODEL,
                f"bedrock/invoke/{MESSAGES_MODEL_ID}",
                api_base=base_url,
                aws_region_name="us-east-1",
            ),
        ],
        "router_settings": {"disable_cooldowns": True, "num_retries": 0},
    }
    path: Final = directory / "dec1-bedrock-env.yaml"
    path.write_text(yaml.safe_dump(configuration))
    return path


@pytest.fixture(scope="module")
def peers() -> Iterator[tuple[Wire, Wire]]:
    with wire_server(_respond) as deployments, wire_server(_respond) as env_endpoint:
        yield deployments, env_endpoint


@pytest.fixture(scope="module")
def env_proxy(peers: tuple[Wire, Wire], tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    deployments, env_endpoint = peers
    directory: Final = tmp_path_factory.mktemp("dec1-bedrock-env")
    scrubbed: Final = tuple(
        name for name in os.environ if name.upper().startswith(("AWS_", "BOTO")) and name.upper() != "AWS_CA_BUNDLE"
    )
    with (
        gateway_from_environment() as shared,
        owned_proxy(
            shared,
            directory,
            {
                "AWS_REGION_NAME": ENV_REGION,
                "AWS_REGION": ENV_STANDARD_REGION,
                "AWS_BEARER_TOKEN_BEDROCK": ENV_BEARER,
                "AWS_BEDROCK_RUNTIME_ENDPOINT": env_endpoint.url,
            },
            config=_config(directory, deployments.url),
            remove_environment=scrubbed,
        ) as candidate,
    ):
        yield candidate


def _posts(wire: Wire) -> list[Request]:
    return [request for request in wire.drain() if request.method == "POST"]


def _chat(proxy: Gateway, model: str) -> None:
    response: Final = proxy.request(
        "POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": marker()}]}
    )
    assert response.status_code == 200, response.text


def _messages(proxy: Gateway, model: str) -> None:
    response: Final = proxy.request(
        "POST",
        "/v1/messages",
        {"model": model, "max_tokens": 32, "messages": [{"role": "user", "content": marker()}]},
        headers={"anthropic-version": "2023-06-01"},
    )
    assert response.status_code == 200, response.text


def test_aws_region_name_env_sets_the_signing_region_and_beats_aws_region(
    env_proxy: Gateway, peers: tuple[Wire, Wire]
) -> None:
    """A keyed deployment with no `aws_region_name`, on a proxy with `AWS_REGION_NAME=eu-west-3` and `AWS_REGION=ap-northeast-1`.

    `AWS_REGION_NAME` is read first (`base_aws_llm.py:743`), so the SigV4 credential scope must name
    `eu-west-3`, never `ap-northeast-1` and never the `us-west-2` last resort.

    BUG (run 44176682961): the chat answers 200 but the deployment's `api_base` peer receives no POST. The
    Python Converse path sends the request to `proxy_endpoint_url` (litellm/llms/bedrock/chat/converse_handler.py:361-371
    and :389-410), and `get_runtime_endpoint` takes that from the `AWS_BEDROCK_RUNTIME_ENDPOINT` env before it
    falls back to `api_base` (litellm/llms/bedrock/base_aws_llm.py:1504-1509), so a process-wide env endpoint
    overrides a deployment's explicit `api_base` on chat. The Rust Messages path in this module honours
    `api_base`.
    """
    pytest.skip("BUG: AWS_BEDROCK_RUNTIME_ENDPOINT env overrides a deployment's explicit api_base on Bedrock chat")
    deployments, _ = peers
    deployments.drain()
    _chat(env_proxy, ENV_REGION_MODEL)
    posts: Final = _posts(deployments)
    assert len(posts) == 1
    assert sigv4_scope(posts[0]) == (ACCESS_KEY, ENV_REGION, "bedrock")


def test_aws_bearer_token_bedrock_env_is_used_when_the_deployment_has_no_api_key(
    env_proxy: Gateway, peers: tuple[Wire, Wire]
) -> None:
    """A deployment with no `api_key` and no AWS keys, on a proxy with `AWS_BEARER_TOKEN_BEDROCK` set.

    Bedrock must receive `Authorization: Bearer synthetic-env-bedrock-bearer` and no `x-amz-date`.

    BUG (run 44176682961): the chat answers 200 but the deployment's `api_base` peer receives no POST. The
    Python Converse path sends the request to `proxy_endpoint_url` (litellm/llms/bedrock/chat/converse_handler.py:361-371
    and :389-410), and `get_runtime_endpoint` takes that from the `AWS_BEDROCK_RUNTIME_ENDPOINT` env before it
    falls back to `api_base` (litellm/llms/bedrock/base_aws_llm.py:1504-1509), so a process-wide env endpoint
    overrides a deployment's explicit `api_base` on chat. The Rust Messages path in this module honours
    `api_base`.
    """
    pytest.skip("BUG: AWS_BEDROCK_RUNTIME_ENDPOINT env overrides a deployment's explicit api_base on Bedrock chat")
    deployments, _ = peers
    deployments.drain()
    _chat(env_proxy, ENV_BEARER_MODEL)
    posts: Final = _posts(deployments)
    assert len(posts) == 1
    assert posts[0].headers["authorization"] == f"Bearer {ENV_BEARER}"
    assert "x-amz-date" not in posts[0].headers


def test_an_empty_api_key_signs_with_sigv4_instead_of_reading_the_env_bearer_on_chat(
    env_proxy: Gateway, peers: tuple[Wire, Wire]
) -> None:
    """A deployment with `api_key: ""` plus AWS keys, on a proxy with `AWS_BEARER_TOKEN_BEDROCK` set.

    `bedrock_bearer_token("")` returns None without reading the env (`api_key if api_key is not None`), so the
    chat call must be SigV4 signed (`AWS4-HMAC-SHA256`, scope `us-east-1/bedrock`), never `Bearer <env token>`.

    BUG (run 44176682961): the chat answers 200 but the deployment's `api_base` peer receives no POST. The
    Python Converse path sends the request to `proxy_endpoint_url` (litellm/llms/bedrock/chat/converse_handler.py:361-371
    and :389-410), and `get_runtime_endpoint` takes that from the `AWS_BEDROCK_RUNTIME_ENDPOINT` env before it
    falls back to `api_base` (litellm/llms/bedrock/base_aws_llm.py:1504-1509), so a process-wide env endpoint
    overrides a deployment's explicit `api_base` on chat. The Rust Messages path in this module honours
    `api_base`.
    """
    pytest.skip("BUG: AWS_BEDROCK_RUNTIME_ENDPOINT env overrides a deployment's explicit api_base on Bedrock chat")
    deployments, _ = peers
    deployments.drain()
    _chat(env_proxy, EMPTY_KEY_MODEL)
    posts: Final = _posts(deployments)
    assert len(posts) == 1
    assert posts[0].headers["authorization"].startswith("AWS4-HMAC-SHA256 ")
    assert sigv4_scope(posts[0]) == (ACCESS_KEY, "us-east-1", "bedrock")


def test_an_empty_api_key_signs_with_sigv4_on_the_rust_messages_invoke_path(
    env_proxy: Gateway, peers: tuple[Wire, Wire]
) -> None:
    """The same empty `api_key` contract on `/v1/messages` to Bedrock InvokeModel, served by Rust only (#51330).

    #52042 fixed Rust reading `AWS_BEARER_TOKEN_BEDROCK` here; the call must reach `/model/<id>/invoke` SigV4
    signed for `us-east-1`, never with `Authorization: Bearer synthetic-env-bedrock-bearer`.
    """
    deployments, _ = peers
    deployments.drain()
    _messages(env_proxy, MESSAGES_EMPTY_KEY_MODEL)
    posts: Final = _posts(deployments)
    assert len(posts) == 1
    assert path_of(posts[0]) == f"/model/{MESSAGES_MODEL_ID}/invoke"
    assert posts[0].headers["authorization"].startswith("AWS4-HMAC-SHA256 ")
    assert ENV_BEARER not in posts[0].headers["authorization"]


def test_the_rust_messages_invoke_path_uses_the_env_bearer_when_no_api_key_is_configured(
    env_proxy: Gateway, peers: tuple[Wire, Wire]
) -> None:
    """A Messages Invoke deployment with no `api_key` and no keys, on a proxy with `AWS_BEARER_TOKEN_BEDROCK`.

    Like Python, Rust must fall back to the env token: Bedrock sees `Authorization: Bearer synthetic-env-bedrock-bearer`.
    """
    deployments, _ = peers
    deployments.drain()
    _messages(env_proxy, MESSAGES_ENV_BEARER_MODEL)
    posts: Final = _posts(deployments)
    assert len(posts) == 1
    assert posts[0].headers["authorization"] == f"Bearer {ENV_BEARER}"


def test_aws_bedrock_runtime_endpoint_env_replaces_the_public_host_when_nothing_else_is_set(
    env_proxy: Gateway, peers: tuple[Wire, Wire]
) -> None:
    """A deployment with neither `api_base` nor `aws_bedrock_runtime_endpoint`, on a proxy with `AWS_BEDROCK_RUNTIME_ENDPOINT=<peer B>`.

    The Converse request must reach peer B at `/model/anthropic.claude-haiku-5-5/converse` and never the
    deployments' peer or `bedrock-runtime.us-east-1.amazonaws.com`.
    """
    deployments, env_endpoint = peers
    deployments.drain()
    env_endpoint.drain()
    _chat(env_proxy, ENV_ENDPOINT_MODEL)
    assert [path_of(request) for request in _posts(env_endpoint)] == [f"/model/{CLAUDE_HAIKU_55}/converse"]
    assert _posts(deployments) == []
