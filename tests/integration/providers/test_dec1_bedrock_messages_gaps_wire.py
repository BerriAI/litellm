"""`/v1/messages` on Bedrock: the cases the `dec1_msgprov` Bedrock files leave open.

Since #51330 (Nov 12) `/v1/messages` on Bedrock InvokeModel is served by the Rust messages crate only, and
#52042 (Nov 16) closed a batch of Rust parity bugs on that path. `messages_endpoint/test_dec1_msgprov_bedrock_invoke.py`
and `test_dec1_msgprov_bedrock_converse.py` already pin the body, betas, SigV4, streaming, errors and spend.
These tests add: the `ca.` cross-region prefix (#52152) on the Messages route, `guardrailConfig` turned into
`X-Amzn-Bedrock-Guardrail*` headers by the Invoke Messages config (`validate_anthropic_messages_environment`,
`litellm/llms/bedrock/messages/invoke_transformations/anthropic_claude_transformation.py:177`; the Rust twin is
`litellm/rust/crates/llms/src/bedrock/messages/invoke_transformations/anthropic_claude_transformation.rs:197`),
its 400 for a config without an identifier, the hoist of a leading system-role turn next to a top-level
`system`, and `guardrailConfig` on the Converse Messages bridge.
"""

from __future__ import annotations

from typing import Final

from integration._support.client import Gateway
from integration.messages_endpoint.dec1_msgprov_support import converse_echo, invoke_echo
from integration.providers.dec1_bedrock_support import (
    CLAUDE_HAIKU_55,
    bedrock,
    body_of,
    error_message,
    marker,
    path_of,
)
from pydantic import JsonValue

INVOKE_ID: Final = "us.anthropic.claude-sonnet-5"
GUARDRAIL: Final[dict[str, JsonValue]] = {
    "guardrailIdentifier": "gr-msg",
    "guardrailVersion": "7",
    "trace": "enabled_full",
}


def test_the_canada_profile_is_kept_in_the_invoke_path_on_the_messages_route(gateway: Gateway) -> None:
    """`bedrock/ca.anthropic.claude-haiku-5-5` on `/v1/messages` is a Claude id, so it is served by InvokeModel.

    The Rust path must post to `/model/ca.anthropic.claude-haiku-5-5/invoke` with the `ca.` prefix intact
    (Rust `CROSS_REGION_INFERENCE_PREFIXES` gained `ca` in #52152) and answer HTTP 200 with an Anthropic message.
    """
    with bedrock(gateway, invoke_echo, model=f"bedrock/ca.{CLAUDE_HAIKU_55}") as rig:
        response: Final = rig.messages(messages=[{"role": "user", "content": marker()}])
        assert response.status_code == 200, response.text
        assert path_of(rig.one_post()) == f"/model/ca.{CLAUDE_HAIKU_55}/invoke"
        assert response.json()["type"] == "message"


def test_deployment_guardrail_config_becomes_guardrail_headers_on_messages_invoke(gateway: Gateway) -> None:
    """A Messages Invoke deployment with `guardrailConfig` `{"guardrailIdentifier": "gr-msg", "guardrailVersion": "7", "trace": "enabled_full"}`.

    Bedrock must see `X-Amzn-Bedrock-GuardrailIdentifier: gr-msg`, `X-Amzn-Bedrock-GuardrailVersion: 7` and
    `X-Amzn-Bedrock-Trace: ENABLED_FULL`, and no `guardrailConfig` key in the Anthropic body, which InvokeModel
    would reject as an extra input.
    """
    with bedrock(gateway, invoke_echo, model=f"bedrock/invoke/{INVOKE_ID}", guardrailConfig=GUARDRAIL) as rig:
        response: Final = rig.messages(messages=[{"role": "user", "content": marker()}])
        assert response.status_code == 200, response.text
        sent: Final = rig.one_post()
        assert sent.headers["x-amzn-bedrock-guardrailidentifier"] == "gr-msg"
        assert sent.headers["x-amzn-bedrock-guardrailversion"] == "7"
        assert sent.headers["x-amzn-bedrock-trace"] == "ENABLED_FULL"
        assert "guardrailConfig" not in body_of(sent)


def test_a_messages_guardrail_config_without_identifier_is_refused_with_400(gateway: Gateway) -> None:
    """A Messages Invoke deployment whose `guardrailConfig` is `{"guardrailVersion": "2"}`.

    Python and Rust both raise a 400 whose message contains `is missing 'guardrailIdentifier'`
    (`bedrock_invoke_guardrail_headers`; Rust test `base_invoke_transformation.rs:163`), and nothing is sent.
    The caller gets the Anthropic error envelope with `type: error`.
    """
    with bedrock(
        gateway, invoke_echo, model=f"bedrock/invoke/{INVOKE_ID}", guardrailConfig={"guardrailVersion": "2"}
    ) as rig:
        response: Final = rig.messages(messages=[{"role": "user", "content": marker()}])
        assert response.status_code == 400, response.text
        assert "is missing 'guardrailIdentifier'" in error_message(response), response.text
        assert rig.posts() == ()


def test_a_top_level_system_and_a_leading_system_turn_are_both_kept_in_order(gateway: Gateway) -> None:
    """A Messages call with `system: "S-top"` and a first message `{"role": "system", "content": "R-turn"}`.

    InvokeModel rejects a system-role message, so it is hoisted, and Bedrock must receive
    `system == [{"type": "text", "text": "S-top"}, {"type": "text", "text": "R-turn"}]` with `messages` starting
    at the user turn (the order #52042 describes as `[S, R]`). UNVERIFIED: whether the top-level text keeps
    any `cache_control` or extra keys.
    """
    with bedrock(gateway, invoke_echo, model=f"bedrock/invoke/{INVOKE_ID}") as rig:
        prompt: Final = marker()
        response: Final = rig.messages(
            system="S-top",
            messages=[{"role": "system", "content": "R-turn"}, {"role": "user", "content": prompt}],
        )
        assert response.status_code == 200, response.text
        sent: Final = body_of(rig.one_post())
        assert sent["system"] == [{"type": "text", "text": "S-top"}, {"type": "text", "text": "R-turn"}], sent
        assert sent["messages"][0]["role"] == "user"  # pyright: ignore[reportIndexIssue, reportArgumentType, reportCallIssue, reportOptionalSubscript]


def test_messages_on_converse_pass_the_deployment_guardrail_config_through(gateway: Gateway) -> None:
    """A `bedrock/converse/` Messages deployment with `guardrailConfig` set.

    The Converse bridge builds a Converse body, where guardrails are a body member, so Bedrock must receive
    `guardrailConfig` equal to the deployment's object at the top level of the `/converse` body.
    UNVERIFIED: that the Anthropic to chat adapter keeps deployment params for the Converse transform.
    """
    with bedrock(gateway, converse_echo, model=f"bedrock/converse/{CLAUDE_HAIKU_55}", guardrailConfig=GUARDRAIL) as rig:
        response: Final = rig.messages(messages=[{"role": "user", "content": marker()}])
        assert response.status_code == 200, response.text
        sent: Final = rig.one_post()
        assert path_of(sent) == f"/model/{CLAUDE_HAIKU_55}/converse"
        assert body_of(sent)["guardrailConfig"] == GUARDRAIL


def test_messages_invoke_region_comes_from_the_deployment_for_sigv4(gateway: Gateway) -> None:
    """A Messages Invoke deployment signed with static keys and `aws_region_name: ca-central-1`, on a `ca.` profile.

    The SigV4 scope must name `ca-central-1` and service `bedrock`: the profile prefix chooses the routing
    geography, never the signing region.
    """
    from integration.providers.dec1_bedrock_support import ACCESS_KEY, sigv4_params, sigv4_scope

    with bedrock(
        gateway,
        invoke_echo,
        model=f"bedrock/invoke/ca.{CLAUDE_HAIKU_55}",
        **sigv4_params(aws_region_name="ca-central-1"),
    ) as rig:
        response: Final = rig.messages(messages=[{"role": "user", "content": marker()}])
        assert response.status_code == 200, response.text
        assert sigv4_scope(rig.one_post()) == (ACCESS_KEY, "ca-central-1", "bedrock")
