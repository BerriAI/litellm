"""Bedrock deployments whose model is an ARN: application inference profiles, prompt routers, provisioned throughput.

Bedrock accepts an ARN anywhere a model id goes. The proxy must put the whole ARN into the request
path as one percent-encoded segment (`encode_model_id`, `litellm/llms/bedrock/base_aws_llm.py:662`), pick
the Converse or Invoke route from what the ARN says (`BedrockModelInfo.get_bedrock_route`,
`litellm/llms/bedrock/common_utils.py:2061`), read the signing region out of the ARN when the deployment
names none, and price the call from the deployment's `base_model` (or its model id when the ARN is passed
as `model_id`), since an ARN names no row in the cost map.

An application inference profile ARN ends in an opaque id, so `is_bedrock_application_inference_profile_arn`
(`common_utils.py:1210`) sends it to Converse. A provisioned throughput ARN also names no provider and so
falls through to Converse (`_names_no_invoke_provider`, `common_utils.py:1870`). A prompt router ARN's tail
names `anthropic.claude`, and `get_bedrock_invoke_provider` (`base_aws_llm.py:561`) finds `anthropic` in it by
substring, so it goes to InvokeModel with the Anthropic body. The Rust router core already resolves
deployments, and the provider transforms are next; each of these id decisions changes the URL Bedrock
sees, and a wrong URL is a 4xx from AWS for every call on that deployment.
"""

from __future__ import annotations

from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request
from integration.providers.dec1_bedrock3_support import (
    APP_PROFILE_ARN,
    PROMPT_ROUTER_ARN,
    PROVISIONED_ARN,
    SONNET_45,
    SONNET_45_INPUT,
    SONNET_45_OUTPUT,
    quoted,
    response_cost,
)
from integration.providers.dec1_bedrock_support import (
    ACCESS_KEY,
    bedrock,
    body_of,
    converse_reply,
    converse_stream_reply,
    json_reply,
    sigv4_params,
    sigv4_scope,
    spend_row,
)

CONVERSE_COST: Final = 11 * SONNET_45_INPUT + 7 * SONNET_45_OUTPUT
"""11 input and 7 output tokens (the default `converse_json` usage) at the Sonnet 4.5 cost map rates."""


def _ok(_: Request) -> Reply:
    return converse_reply()


def _claude_invoke_reply(_: Request) -> Reply:
    return json_reply(
        {
            "id": "msg_bdrk_router",
            "type": "message",
            "role": "assistant",
            "model": SONNET_45,
            "content": [{"type": "text", "text": "routed"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 12, "output_tokens": 6},
        }
    )


def test_an_application_inference_profile_arn_model_posts_to_converse_with_the_whole_arn_encoded(
    gateway: Gateway,
) -> None:
    """A deployment `bedrock/arn:aws:bedrock:us-west-2:<acct>:application-inference-profile/dec1prof3abc` is called.

    The ARN names no provider, so the call must go to Converse: exactly one
    `POST /model/<ARN with ':' and '/' percent-encoded>/converse`, never `/invoke`. Encoding the `/` keeps the
    ARN one path segment; a rewrite that sends it raw splits the path and Bedrock answers 404.
    """
    with bedrock(gateway, _ok, model=f"bedrock/{APP_PROFILE_ARN}") as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        assert rig.one_post().target == f"/model/{quoted(APP_PROFILE_ARN)}/converse"


def test_a_streamed_call_on_an_application_inference_profile_uses_converse_stream(gateway: Gateway) -> None:
    """The same application inference profile deployment is called with `stream: true`.

    Bedrock must receive `POST /model/<encoded ARN>/converse-stream` (`converse_handler.py:367`) and the
    caller gets a 200 SSE answer.
    """
    with bedrock(gateway, lambda _: converse_stream_reply(), model=f"bedrock/{APP_PROFILE_ARN}") as rig:
        response: Final = rig.chat(stream=True)
        assert response.status_code == 200, response.text
        assert rig.one_post().target == f"/model/{quoted(APP_PROFILE_ARN)}/converse-stream"


def test_the_region_inside_an_application_profile_arn_model_is_the_signing_region(gateway: Gateway) -> None:
    """The application inference profile ARN (region `us-west-2`) is the model, with SigV4 keys and no `aws_region_name`.

    `_get_aws_region_name` falls back to `_get_aws_region_from_model_arn(model)` (`base_aws_llm.py:737`),
    so the SigV4 credential scope must name `us-west-2` and the `bedrock` service.
    """
    with bedrock(gateway, _ok, model=f"bedrock/{APP_PROFILE_ARN}", **sigv4_params(aws_region_name=None)) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        assert sigv4_scope(rig.one_post()) == (ACCESS_KEY, "us-west-2", "bedrock")


def test_an_application_profile_is_priced_from_the_deployment_base_model(gateway: Gateway) -> None:
    """The profile deployment sets `model_info.base_model: bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0` and no prices.

    The ARN names no cost map row, so the proxy must price from `base_model`: 11 input and 7 output tokens
    at 3e-06 and 1.5e-05 per token, so `x-litellm-response-cost` is 0.000138.
    UNVERIFIED: whether `base_model` must carry the `bedrock/` prefix; see `_get_base_model_from_metadata`.
    """
    with bedrock(
        gateway, _ok, model=f"bedrock/{APP_PROFILE_ARN}", model_info={"base_model": f"bedrock/{SONNET_45}"}
    ) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        assert response_cost(response) == pytest.approx(CONVERSE_COST)


def test_an_application_profile_spend_row_keeps_the_deployment_model_and_the_base_model_price(gateway: Gateway) -> None:
    """Same profile deployment with `base_model`; the SpendLogs row for the call is read back.

    The row must carry `custom_llm_provider: bedrock`, `spend` 0.000138 (the base model rate) and a non
    empty `model_id` naming the deployment, so per-deployment spend still adds up when the ARN is opaque.
    """
    with bedrock(
        gateway, _ok, model=f"bedrock/{APP_PROFILE_ARN}", model_info={"base_model": f"bedrock/{SONNET_45}"}
    ) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        row: Final = spend_row(response.json()["id"])
        assert row["custom_llm_provider"] == "bedrock", row
        assert float(row["spend"]) == pytest.approx(CONVERSE_COST), row  # pyright: ignore[reportArgumentType]
        assert row["model_id"], row


def test_a_provisioned_throughput_arn_model_goes_to_converse(gateway: Gateway) -> None:
    """A deployment `bedrock/arn:aws:bedrock:us-east-1:<acct>:provisioned-model/dec1pt9xyz` is called.

    The tail is opaque and names no Invoke provider, so `_names_no_invoke_provider` sends it to Converse:
    one `POST /model/<encoded ARN>/converse`.
    """
    with bedrock(gateway, _ok, model=f"bedrock/{PROVISIONED_ARN}") as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        assert rig.one_post().target == f"/model/{quoted(PROVISIONED_ARN)}/converse"


def test_a_provisioned_throughput_model_id_is_priced_from_the_deployment_model(gateway: Gateway) -> None:
    """A deployment `bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0` with `model_id` set to the provisioned ARN.

    The URL must carry the encoded provisioned ARN (`converse_handler.py:307`), while the price comes from
    the deployment's model: 11 in and 7 out tokens at the Sonnet 4.5 row, so the cost header is 0.000138.
    """
    with bedrock(gateway, _ok, model=f"bedrock/{SONNET_45}", model_id=PROVISIONED_ARN) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        assert rig.one_post().target == f"/model/{quoted(PROVISIONED_ARN)}/converse"
        assert response_cost(response) == pytest.approx(CONVERSE_COST)


def test_a_provisioned_model_id_never_leaks_into_the_converse_body(gateway: Gateway) -> None:
    """`model_id` is a routing parameter only.

    Bedrock's Converse body must hold no `model_id` or `modelId` key at any top level position; the
    handler pops it before building the body (`converse_handler.py:304`).
    """
    with bedrock(gateway, _ok, model=f"bedrock/{SONNET_45}", model_id=PROVISIONED_ARN) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        sent: Final = body_of(rig.one_post())
        assert not {"model_id", "modelId"} & set(sent), sent


def test_a_prompt_router_arn_goes_to_invoke_with_the_anthropic_body(gateway: Gateway) -> None:
    """A deployment `bedrock/arn:aws:bedrock:us-east-1:<acct>:default-prompt-router/anthropic.claude:1` is called.

    `get_bedrock_invoke_provider` finds `anthropic` in the ARN by substring, so the route is InvokeModel:
    one `POST /model/<encoded ARN>/invoke` whose body carries `anthropic_version: bedrock-2023-05-31` and an
    Anthropic `messages` list, and the Anthropic answer `routed` comes back as the chat content.
    UNVERIFIED: read from `base_aws_llm.py:561` and `common_utils.py:2078`; Converse would also accept a router.
    """
    with bedrock(gateway, _claude_invoke_reply, model=f"bedrock/{PROMPT_ROUTER_ARN}") as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        sent: Final = rig.one_post()
        assert sent.target == f"/model/{quoted(PROMPT_ROUTER_ARN)}/invoke", sent.target
        assert body_of(sent)["anthropic_version"] == "bedrock-2023-05-31"
        assert response.json()["choices"][0]["message"]["content"] == "routed"


def test_a_prompt_router_arn_pinned_to_converse_posts_to_converse(gateway: Gateway) -> None:
    """The same prompt router ARN behind the explicit `bedrock/converse/` route.

    The `converse/` prefix wins over provider detection (`named_bedrock_route`), so Bedrock must receive
    `POST /model/<encoded ARN>/converse` with a Converse body (`messages[0].content[0].text`).
    """
    with bedrock(gateway, _ok, model=f"bedrock/converse/{PROMPT_ROUTER_ARN}") as rig:
        response: Final = rig.chat("router prompt")
        assert response.status_code == 200, response.text
        sent: Final = rig.one_post()
        assert sent.target == f"/model/{quoted(PROMPT_ROUTER_ARN)}/converse", sent.target
        assert body_of(sent)["messages"] == [{"role": "user", "content": [{"text": "router prompt"}]}]


def test_an_application_profile_model_id_on_the_invoke_route_is_encoded_into_the_invoke_path(gateway: Gateway) -> None:
    """A `bedrock/invoke/anthropic.claude-sonnet-4-5-20250929-v1:0` deployment with `model_id` = the profile ARN.

    `get_bedrock_model_id` pops `model_id` and percent-encodes it (`base_aws_llm.py:606`), so Bedrock must
    receive `POST /model/<encoded profile ARN>/invoke` with the Anthropic body (`anthropic_version` set), and
    the profile ARN's region `us-west-2` must sign the request when the deployment names no region.
    """
    with bedrock(
        gateway,
        _claude_invoke_reply,
        model=f"bedrock/invoke/{SONNET_45}",
        model_id=APP_PROFILE_ARN,
        **sigv4_params(aws_region_name=None),
    ) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        sent: Final = rig.one_post()
        assert sent.target == f"/model/{quoted(APP_PROFILE_ARN)}/invoke", sent.target
        assert body_of(sent)["anthropic_version"] == "bedrock-2023-05-31"
        assert sigv4_scope(sent)[1] == "us-west-2"
