"""Tag budgets (``litellm_settings.tag_budget_config``, an enterprise feature) on a two-worker proxy.

Tags are per request, so one uncapped deployment serves every cell and each cell owns its tags. Probes carry
``PROVIDER_FAILURE`` and are never charged.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final

import pytest
from integration._support.client import Gateway, gateway_from_environment
from integration._support.wire import wire_server
from integration.routing.router_budgets import _rig as rig
from pydantic import JsonValue

ISSUE_43214: Final = "https://github.com/BerriAI/litellm/issues/43214"
MODEL: Final = "tagged"


@dataclass(frozen=True, slots=True)
class TagRig:
    gateway: Gateway
    budget: rig.BudgetRig


@pytest.fixture(scope="module")
def tags(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TagRig]:
    tmp_path: Final = tmp_path_factory.mktemp("tag-budgets")
    with gateway_from_environment() as gateway, wire_server(rig.upstream) as upstream:
        config: Final = rig.write_config(
            tmp_path / "tags.yaml",
            (rig.deployment(MODEL, "hosted_vllm/tagged", upstream.url, model_id="tagged"),),
            tag_budget_config={
                "tag-at-cap": {"max_budget": 2 * rig.CALL_COST, "budget_duration": "1d"},
                "tag-tiny": {"max_budget": 0.01, "budget_duration": "1d"},
                "tag-zero": {"max_budget": 0, "budget_duration": "1d"},
                "tag-uncapped": {"budget_duration": "1d"},
                "tag-no-duration": {"max_budget": 0.01},
                "tag-roomy": {"max_budget": 10, "budget_duration": "1d"},
                "tag-multi": {"max_budget": 0.01, "budget_duration": "1d"},
                "tag-header": {"max_budget": 0.01, "budget_duration": "1d"},
            },
        )
        with rig.budget_proxy(gateway, tmp_path, config) as budget:
            yield TagRig(budget.gateway, budget)


def _tagged(*names: str) -> dict[str, JsonValue]:
    return {"metadata": {"tags": list(names)}}


def test_spend_over_a_tiny_tag_cap_rejects_requests_carrying_that_tag(tags: TagRig) -> None:
    first: Final = rig.chat(tags.gateway, MODEL, "tag tiny first", _tagged("tag-tiny"))
    assert first.status_code == 200, first.text

    blocked: Final = rig.until_rejected(tags.gateway, MODEL, rig.probe_text("tag tiny"), _tagged("tag-tiny"))

    assert blocked.status_code == 429, blocked.text
    assert f"Exceeded budget for tag='tag-tiny', tag_spend={rig.CALL_COST}, tag_budget_limit=0.01" in blocked.text


def test_spend_reaching_exactly_the_tag_cap_blocks_the_next_request(tags: TagRig) -> None:
    first: Final = rig.chat(tags.gateway, MODEL, "tag at cap first", _tagged("tag-at-cap"))
    second: Final = rig.chat(tags.gateway, MODEL, "tag at cap second", _tagged("tag-at-cap"))
    assert (first.status_code, second.status_code) == (200, 200), (first.text, second.text)

    blocked: Final = rig.until_rejected(tags.gateway, MODEL, rig.probe_text("tag at cap"), _tagged("tag-at-cap"))

    assert f"tag_spend={2 * rig.CALL_COST}, tag_budget_limit={2 * rig.CALL_COST}" in blocked.text
    assert tags.budget.settled("tag_spend:tag-at-cap:1d", 2 * rig.CALL_COST) == 2 * rig.CALL_COST


@pytest.mark.xfail(strict=True, reason=f"max_budget 0 is treated as unlimited: {ISSUE_43214}")
def test_a_zero_tag_cap_rejects_the_first_request(tags: TagRig) -> None:
    response: Final = rig.chat(tags.gateway, MODEL, rig.probe_text("tag zero"), _tagged("tag-zero"))

    assert rig.is_budget_rejection(response), response.text


def test_a_tag_without_max_budget_stays_uncapped(tags: TagRig) -> None:
    responses: Final = tuple(
        rig.chat(tags.gateway, MODEL, f"tag uncapped {index}", _tagged("tag-uncapped")) for index in range(3)
    )

    assert all(response.status_code == 200 for response in responses), [response.text for response in responses]
    assert tags.budget.settled("tag_spend:tag-uncapped:1d", 3 * rig.CALL_COST) == 3 * rig.CALL_COST


@pytest.mark.xfail(strict=True, reason="a tag max_budget without budget_duration is never counted or enforced")
def test_a_tag_cap_without_budget_duration_is_still_enforced(tags: TagRig) -> None:
    first: Final = rig.chat(tags.gateway, MODEL, "tag without duration", _tagged("tag-no-duration"))
    assert first.status_code == 200, first.text

    rig.until_rejected(tags.gateway, MODEL, rig.probe_text("tag no duration"), _tagged("tag-no-duration"), seconds=10)


def test_two_tags_are_each_charged_once_and_only_the_exhausted_one_blocks(tags: TagRig) -> None:
    first: Final = rig.chat(tags.gateway, MODEL, "two tags", _tagged("tag-roomy", "tag-multi"))
    assert first.status_code == 200, first.text

    blocked: Final = rig.until_rejected(
        tags.gateway, MODEL, rig.probe_text("two tags"), _tagged("tag-roomy", "tag-multi")
    )

    assert "tag='tag-multi'" in blocked.text and "tag='tag-roomy'" not in blocked.text, blocked.text
    assert tags.budget.settled("tag_spend:tag-multi:1d", rig.CALL_COST) == rig.CALL_COST
    assert tags.budget.settled("tag_spend:tag-roomy:1d", rig.CALL_COST) == rig.CALL_COST
    roomy_only: Final = rig.chat(tags.gateway, MODEL, rig.probe_text("roomy only"), _tagged("tag-roomy"))
    untagged: Final = rig.chat(tags.gateway, MODEL, rig.probe_text("untagged"))
    assert (roomy_only.status_code, untagged.status_code) == (500, 500), (roomy_only.text, untagged.text)


def test_a_tag_spent_through_the_header_blocks_the_same_tag_in_body_metadata(tags: TagRig) -> None:
    first: Final = tags.gateway.request(
        "POST",
        "/v1/chat/completions",
        rig.body_for("chat", MODEL, "tag header"),
        headers={"x-litellm-tags": "tag-header"},
    )
    assert first.status_code == 200, first.text

    blocked: Final = rig.until_rejected(tags.gateway, MODEL, rig.probe_text("tag header"), _tagged("tag-header"))

    assert "tag='tag-header'" in blocked.text
