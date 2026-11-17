"""Azure data zone deployments (EU, US) are priced through `base_model`, not through the URL.

A Data Zone Standard deployment is an ordinary Azure deployment on the wire: same host shape, same
`/openai/deployments/<name>/chat/completions` path. Only the price differs, about 10% above Global Standard, and
the cost map carries it in `azure/eu/<model>` and `azure/us/<model>` rows. The proxy can only know a deployment is
data zone when the operator names such a row as `base_model` (in `model_info` or `litellm_params`); without it the
call is priced from the model Azure reports. Expected values are computed from model_prices_and_context_window.json:

- `azure/eu/gpt-4o-2024-08-06` and `azure/us/gpt-4o-2024-08-06`: 2.75e-06 input, 1.1e-05 output, 1.375e-06 cached
- `azure/gpt-4o-2024-08-06` and `azure/global/gpt-4o-2024-08-06`: 2.5e-06 input, 1e-05 output
- `azure/eu/gpt-5-nano-2025-08-07`: 5.5e-08 input, 4.4e-07 output

The peer reports 11 prompt and 7 completion tokens unless a test says otherwise. Pricing moves to Rust, and a
lookup that ignores `base_model` would bill every EU deployment at the global rate.
"""

from __future__ import annotations

from typing import Final

import pytest
from integration._support.client import Gateway
from integration.providers.dec1_azure_support import always, azure, chat_reply, path_of, spend_row

GPT4O: Final = "gpt-4o-2024-08-06"
EU_GPT4O: Final = 11 * 2.75e-06 + 7 * 1.1e-05
GLOBAL_GPT4O: Final = 11 * 2.5e-06 + 7 * 1e-05


@pytest.mark.parametrize("zone", ["eu", "us"])
def test_a_data_zone_base_model_in_model_info_prices_at_the_zone_rate(gateway: Gateway, zone: str) -> None:
    """An `azure/dz-gpt4o` deployment with `model_info.base_model: azure/<zone>/gpt-4o-2024-08-06`; Azure reports
    `model: gpt-4o-2024-08-06`.

    The SpendLogs row must have `spend == 11 * 2.75e-06 + 7 * 1.1e-05 = 0.00010725`, the data zone rate, not the
    global `9.75e-05`.
    """
    with azure(
        gateway,
        always(chat_reply("pong", model=GPT4O)),
        model="azure/dz-gpt4o",
        model_info={"base_model": f"azure/{zone}/{GPT4O}"},
    ) as rig:
        body: Final = rig.chat().json()
        assert float(str(spend_row(body["id"])["spend"])) == pytest.approx(EU_GPT4O)


def test_a_data_zone_base_model_in_litellm_params_prices_the_same(gateway: Gateway) -> None:
    """The EU base model is set as `litellm_params.base_model` instead of in `model_info`.

    `get_base_model_from_metadata` reads `litellm_params.base_model` first, so spend must again be `0.00010725`.
    """
    with azure(
        gateway, always(chat_reply("pong", model=GPT4O)), model="azure/dz-gpt4o", base_model=f"azure/eu/{GPT4O}"
    ) as rig:
        body: Final = rig.chat().json()
        assert float(str(spend_row(body["id"])["spend"])) == pytest.approx(EU_GPT4O)


def test_a_global_base_model_prices_at_the_global_rate(gateway: Gateway) -> None:
    """`model_info.base_model: azure/global/gpt-4o-2024-08-06`.

    Spend must be `11 * 2.5e-06 + 7 * 1e-05 = 9.75e-05`, so the three deployment types stay distinguishable.
    """
    with azure(
        gateway,
        always(chat_reply("pong", model=GPT4O)),
        model="azure/dz-gpt4o",
        model_info={"base_model": f"azure/global/{GPT4O}"},
    ) as rig:
        body: Final = rig.chat().json()
        assert float(str(spend_row(body["id"])["spend"])) == pytest.approx(GLOBAL_GPT4O)


def test_without_base_model_a_data_zone_deployment_is_billed_at_the_reported_models_global_rate(
    gateway: Gateway,
) -> None:
    """An EU deployment named `eu-gpt4o` with no base_model; Azure reports `model: gpt-4o-2024-08-06`.

    Nothing tells the proxy the deployment is data zone, so spend must be the `azure/gpt-4o-2024-08-06` rate
    `9.75e-05`. Pinned so the under-billing is a visible, deliberate behaviour. UNVERIFIED: that Azure cost
    lookup uses the response model when no base_model is set.
    """
    with azure(gateway, always(chat_reply("pong", model=GPT4O)), model="azure/eu-gpt4o") as rig:
        body: Final = rig.chat().json()
        assert float(str(spend_row(body["id"])["spend"])) == pytest.approx(GLOBAL_GPT4O)


def test_cached_prompt_tokens_use_the_data_zone_cache_read_rate(gateway: Gateway) -> None:
    """EU base model; Azure reports 11 prompt tokens of which 4 cached, and 7 completion tokens.

    Spend must be `7 * 2.75e-06 + 4 * 1.375e-06 + 7 * 1.1e-05 = 0.00010175`.
    """
    usage: Final = {
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
        "prompt_tokens_details": {"cached_tokens": 4},
    }
    with azure(
        gateway,
        always(chat_reply("pong", model=GPT4O, usage=usage)),
        model="azure/dz-gpt4o",
        model_info={"base_model": f"azure/eu/{GPT4O}"},
    ) as rig:
        body: Final = rig.chat().json()
        assert float(str(spend_row(body["id"])["spend"])) == pytest.approx(7 * 2.75e-06 + 4 * 1.375e-06 + 7 * 1.1e-05)


def test_an_eu_gpt5_nano_base_model_prices_at_its_zone_row(gateway: Gateway) -> None:
    """A GPT-5 deployment `azure/dz-nano` with `model_info.base_model: azure/eu/gpt-5-nano-2025-08-07`.

    Spend must be `11 * 5.5e-08 + 7 * 4.4e-07 = 3.685e-06`.
    """
    with azure(
        gateway,
        always(chat_reply("pong", model="gpt-5-nano-2025-08-07")),
        model="azure/dz-nano",
        model_info={"base_model": "azure/eu/gpt-5-nano-2025-08-07"},
    ) as rig:
        body: Final = rig.chat().json()
        assert float(str(spend_row(body["id"])["spend"])) == pytest.approx(11 * 5.5e-08 + 7 * 4.4e-07)


def test_a_data_zone_base_model_never_changes_the_azure_url(gateway: Gateway) -> None:
    """EU base model on the `azure/dz-gpt4o` deployment.

    The base model is a pricing hint only: Azure must still see `POST /openai/deployments/dz-gpt4o/chat/completions`,
    never a path containing `eu/` or the base model name.
    """
    with azure(
        gateway,
        always(chat_reply("pong", model=GPT4O)),
        model="azure/dz-gpt4o",
        model_info={"base_model": f"azure/eu/{GPT4O}"},
    ) as rig:
        rig.chat()
        assert path_of(rig.one_post()) == "/openai/deployments/dz-gpt4o/chat/completions"
