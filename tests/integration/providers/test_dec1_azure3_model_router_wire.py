"""Azure AI Foundry Model Router: the deployment Azure is asked for differs from the model that answers.

A Model Router deployment is detected by name (`AzureFoundryModelInfo.get_azure_ai_route`): an
`azure_ai/model_router/<deployment>` prefix, or any model containing `model-router` / `model_router`. It is
served by `AzureModelRouterConfig` (`litellm/llms/azure_ai/azure_model_router/transformation.py`):

- the request body `model` is the deployment with the routing prefix stripped (`get_base_model`);
- the response `model` becomes `azure_ai/<model Azure reports>`, i.e. the model the router picked, and that
  name is stamped into hidden params so spend logs record it;
- cost is the picked model's own token price plus the router's flat fee, `azure_ai/model_router`
  `input_cost_per_token: 1.4e-07` per prompt token, added exactly once (`azure_ai/cost_calculator.py`).

`spend/test_model_router_selected_model.py` already pins the plain `model_router/<name>` response and SpendLogs
model. This file adds the name-detected router, streaming, and the fee arithmetic. Pricing logic is moving to
Rust, and the "deployment != reported model" split is exactly what a naive port would collapse.
"""

from __future__ import annotations

from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Request
from integration.providers.dec1_azure_support import (
    always,
    body_of,
    chat_json,
    chat_reply,
    chat_stream,
    foundry,
    json_reply,
    path_of,
    spend_row,
    sse_events,
)

ROUTER_DEPLOYMENT: Final = "dec1-router"
SELECTED: Final = "Phi-4"
"""Routed-to model with an `azure_ai/Phi-4` cost map row: 1.25e-07 input, 5e-07 output per token."""
ROUTER_FEE_PER_PROMPT_TOKEN: Final = 1.4e-07
USAGE_20_20: Final = {"prompt_tokens": 20, "completion_tokens": 20, "total_tokens": 40}


def _routed(_: Request):  # noqa: ANN202
    return json_reply(chat_json("routed answer", model=SELECTED, usage=USAGE_20_20))


def test_a_name_detected_router_sends_its_full_deployment_name_as_the_body_model(gateway: Gateway) -> None:
    """An `azure_ai/model-router` deployment (no `model_router/` prefix, detected by the name).

    There is no prefix to strip, so the Foundry body `model` must be exactly `model-router` and the call must
    go to `POST /chat/completions` under the loopback `api_base`.
    """
    with foundry(gateway, _routed, model="azure_ai/model-router") as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        request: Final = rig.one_post()
        assert (path_of(request), body_of(request)["model"]) == ("/chat/completions", "model-router")


def test_the_prefixed_router_strips_model_router_from_the_body_model(gateway: Gateway) -> None:
    """An `azure_ai/model_router/dec1-router` deployment.

    The body `model` must be `dec1-router`, never `model_router/dec1-router` (`strip_model_router_prefix`).
    """
    with foundry(gateway, _routed, model=f"azure_ai/model_router/{ROUTER_DEPLOYMENT}") as rig:
        rig.chat()
        assert body_of(rig.one_post())["model"] == ROUTER_DEPLOYMENT


def test_the_caller_sees_the_model_the_router_picked(gateway: Gateway) -> None:
    """Azure reports `model: Phi-4` for a call to the `azure_ai/model-router` deployment.

    The caller's answer `model` must be `azure_ai/Phi-4`, not the deployment name and not the public group.
    """
    with foundry(gateway, _routed, model="azure_ai/model-router") as rig:
        assert rig.chat().json()["model"] == f"azure_ai/{SELECTED}"


def test_the_spend_row_records_the_picked_model_and_the_router_group(gateway: Gateway) -> None:
    """Same routed call.

    The one SpendLogs row must have `model == "azure_ai/Phi-4"`, `model_group` the public group name and
    `custom_llm_provider == "azure_ai"`.
    """
    with foundry(gateway, _routed, model="azure_ai/model-router") as rig:
        body: Final = rig.chat().json()
        row: Final = spend_row(body["id"])
        assert (row["model"], row["model_group"], row["custom_llm_provider"]) == (
            f"azure_ai/{SELECTED}",
            rig.model,
            "azure_ai",
        )


def test_the_spend_adds_the_router_fee_once_to_the_picked_models_price(gateway: Gateway) -> None:
    """Azure reports 20 prompt and 20 completion tokens served by `Phi-4`.

    Spend must be the Phi-4 price plus the router fee once:
    `20 * 1.25e-07 + 20 * 5e-07 + 20 * 1.4e-07 = 1.53e-05`. UNVERIFIED: whether the fee enters through
    `cost_per_token(request_model=...)` or `calculate_additional_costs`; the docstring of
    `azure_ai/cost_calculator.py:cost_per_token` promises exactly once, which is what this pins.
    """
    with foundry(gateway, _routed, model="azure_ai/model-router") as rig:
        body: Final = rig.chat().json()
        row: Final = spend_row(body["id"])
        assert float(str(row["spend"])) == pytest.approx(20 * 1.25e-07 + 20 * 5e-07 + 20 * ROUTER_FEE_PER_PROMPT_TOKEN)


def test_a_router_answer_naming_the_router_itself_is_billed_at_the_fee_alone(gateway: Gateway) -> None:
    """Azure reports `model: model-router` (no routed model surfaced) with 20 prompt and 20 completion tokens.

    A response priced as the router entry already carries the fee (`is_router_fee_entry`), so spend must be
    `20 * 1.4e-07 + 20 * 0 = 2.8e-06`, with no second fee on top.
    """
    reply: Final = chat_reply("x", model="model-router", usage=USAGE_20_20)
    with foundry(gateway, always(reply), model="azure_ai/model-router") as rig:
        body: Final = rig.chat().json()
        assert float(str(spend_row(body["id"])["spend"])) == pytest.approx(20 * ROUTER_FEE_PER_PROMPT_TOKEN)


def test_a_streamed_router_call_strips_the_prefix_and_asks_to_stream(gateway: Gateway) -> None:
    """The caller streams through `azure_ai/model_router/dec1-router`.

    Foundry must get one POST whose body has `model: dec1-router` and `stream: true`, and the caller must get
    the streamed text `routed` ending in `[DONE]`.
    """
    with foundry(
        gateway, always(chat_stream(["rou", "ted"])), model=f"azure_ai/model_router/{ROUTER_DEPLOYMENT}"
    ) as rig:
        response: Final = rig.chat(stream=True)
        assert response.status_code == 200, response.text
        body: Final = body_of(rig.one_post())
        assert (body["model"], body["stream"]) == (ROUTER_DEPLOYMENT, True)
        assert (
            "".join(
                str(c.get("delta", {}).get("content") or "")
                for e in sse_events(response)
                for c in (e.get("choices") or [])  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportAttributeAccessIssue, reportUnknownArgumentType]
            )
            == "routed"
        )
