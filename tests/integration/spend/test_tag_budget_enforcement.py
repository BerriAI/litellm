import os
import uuid
from pathlib import Path
from typing import Final

import httpx
from integration._support.client import Gateway, eventually, string_value
from integration._support.process import owned_proxy
from redis import Redis


def test_spend_over_a_tag_max_budget_rejects_the_next_request(gateway: Gateway) -> None:
    tag: Final = f"tag-budget-{uuid.uuid4().hex}"

    def delete_tag() -> None:
        gateway.post("/tag/delete", {"name": tag})

    with gateway.scenario() as scenario:
        model: Final = scenario.model(input_cost_per_token=0.01, output_cost_per_token=0.01)
        gateway.post("/tag/new", {"name": tag, "max_budget": 0.0001})
        scenario.cleanups.callback(delete_tag)
        first: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": f"tag spend {tag}"}],
                "metadata": {"tags": [tag]},
            },
        )
        assert first.status_code == 200, first.text

        def rejection() -> int:
            return gateway.request(
                "POST",
                "/v1/chat/completions",
                {
                    "model": model,
                    "messages": [{"role": "user", "content": f"tag budget probe {tag}"}],
                    "metadata": {"tags": [tag]},
                },
            ).status_code

        status: Final = eventually(rejection, lambda code: code != 200, seconds=70)
        assert status in (400, 422, 429), status
        blocked: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": f"tag budget probe {tag}"}],
                "metadata": {"tags": [tag]},
            },
        )
        assert "budget" in blocked.text.lower(), blocked.text
        control: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": f"untagged probe {tag}"}],
                "metadata": {"tags": [f"other-{tag}"]},
            },
        )
        assert control.status_code == 200, control.text


def test_tag_object_rpm_limit_rejects_the_second_request_across_keys(gateway: Gateway) -> None:
    tag: Final = f"tag-rpm-{uuid.uuid4().hex}"

    def delete_tag() -> None:
        gateway.post("/tag/delete", {"name": tag})

    with gateway.scenario() as scenario:
        model: Final = scenario.model(input_cost_per_token=0.01, output_cost_per_token=0.01)
        key_a: Final = scenario.key()
        key_b: Final = scenario.key()
        gateway.post("/tag/new", {"name": tag, "rpm_limit": 1})
        scenario.cleanups.callback(delete_tag)
        first: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": f"tag rpm {tag}"}],
                "metadata": {"tags": [tag]},
            },
            key=key_a,
        )
        assert first.status_code == 200, first.text
        second: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": f"tag rpm {tag}"}],
                "metadata": {"tags": [tag]},
            },
            key=key_b,
        )
        assert second.status_code == 429, second.text
        assert "tag" in second.text.lower(), second.text
        control: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": f"other tag rpm {tag}"}],
                "metadata": {"tags": [f"other-{tag}"]},
            },
            key=key_b,
        )
        assert control.status_code == 200, control.text


def test_tag_object_rpm_limit_is_shared_across_teams_organizations_and_users(gateway: Gateway) -> None:
    tag: Final = f"tag-rpm-{uuid.uuid4().hex}"

    def delete_tag() -> None:
        gateway.post("/tag/delete", {"name": tag})

    def delete_organization(identity: str) -> None:
        deleted: Final = gateway.request("DELETE", "/organization/delete", {"organization_ids": [identity]})
        assert deleted.status_code == 200, deleted.text

    with gateway.scenario() as scenario:
        model: Final = scenario.model(input_cost_per_token=0.01, output_cost_per_token=0.01)
        org_a: Final = gateway.post("/organization/new", {"organization_alias": f"integration-{uuid.uuid4().hex}"})
        org_b: Final = gateway.post("/organization/new", {"organization_alias": f"integration-{uuid.uuid4().hex}"})
        org_a_id: Final = string_value(org_a["organization_id"])
        org_b_id: Final = string_value(org_b["organization_id"])
        scenario.cleanups.callback(delete_organization, org_a_id)
        scenario.cleanups.callback(delete_organization, org_b_id)
        team_a: Final = scenario.team(organization_id=org_a_id)
        team_b: Final = scenario.team(organization_id=org_b_id)
        user_1: Final = scenario.user()
        user_2: Final = scenario.user()
        key_team_a: Final = scenario.key(team_id=team_a)
        key_team_b: Final = scenario.key(team_id=team_b)
        key_user_1: Final = scenario.key(user_id=user_1)
        key_user_2: Final = scenario.key(user_id=user_2)
        gateway.post("/tag/new", {"name": tag, "rpm_limit": 3})
        scenario.cleanups.callback(delete_tag)

        def tagged_request(key: str, request_tag: str) -> httpx.Response:
            return gateway.request(
                "POST",
                "/v1/chat/completions",
                {
                    "model": model,
                    "messages": [{"role": "user", "content": f"tag rpm {request_tag}"}],
                    "metadata": {"tags": [request_tag]},
                },
                key=key,
            )

        for scoped_key in (key_team_a, key_team_b, key_user_1):
            admitted: Final = tagged_request(scoped_key, tag)
            assert admitted.status_code == 200, admitted.text
        blocked: Final = tagged_request(key_user_2, tag)
        assert blocked.status_code == 429, blocked.text
        assert "tag" in blocked.text.lower(), blocked.text
        control: Final = tagged_request(key_user_2, f"other-{tag}")
        assert control.status_code == 200, control.text


def test_tag_object_tpm_limit_rejects_the_next_request_once_tokens_are_charged(gateway: Gateway) -> None:
    tag: Final = f"tag-tpm-{uuid.uuid4().hex}"

    def delete_tag() -> None:
        gateway.post("/tag/delete", {"name": tag})

    with gateway.scenario() as scenario:
        model: Final = scenario.model(input_cost_per_token=0.01, output_cost_per_token=0.01)
        key: Final = scenario.key()
        gateway.post("/tag/new", {"name": tag, "tpm_limit": 39})
        scenario.cleanups.callback(delete_tag)
        first: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": f"tag tpm {tag}"}],
                "metadata": {"tags": [tag]},
            },
            key=key,
        )
        assert first.status_code == 200, first.text

        with Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"])) as cache:
            charged: Final = eventually(
                lambda: int(cache.get(f"{{tag:{tag}}}:tokens") or 0),
                lambda tokens: tokens >= 40,
                seconds=70,
            )
        assert charged >= 40, charged
        second: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": f"tag tpm {tag}"}],
                "metadata": {"tags": [tag]},
            },
            key=key,
        )
        assert second.status_code == 429, second.text
        assert "tag" in second.text.lower(), second.text
        assert "token" in second.text.lower(), second.text
        control: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": f"other tag tpm {tag}"}],
                "metadata": {"tags": [f"other-{tag}"]},
            },
            key=key,
        )
        assert control.status_code == 200, control.text


def test_key_tag_rpm_limit_rejects_the_second_request_carrying_that_tag(gateway: Gateway) -> None:
    tag: Final = f"tag-rpm-{uuid.uuid4().hex}"
    with gateway.scenario() as scenario:
        model: Final = scenario.model(input_cost_per_token=0.01, output_cost_per_token=0.01)
        key: Final = scenario.key(metadata={"tag_rpm_limit": {tag: 1}})
        first: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": f"tag rpm {tag}"}],
                "metadata": {"tags": [tag]},
            },
            key=key,
        )
        assert first.status_code == 200, first.text
        second: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": f"tag rpm {tag}"}],
                "metadata": {"tags": [tag]},
            },
            key=key,
        )
        assert second.status_code == 429, second.text
        assert "rpm" in second.text.lower() or "rate" in second.text.lower(), second.text
        control: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": f"other tag rpm {tag}"}],
                "metadata": {"tags": [f"other-{tag}"]},
            },
            key=key,
        )
        assert control.status_code == 200, control.text


def test_tag_budget_duration_resets_spend_and_unblocks_the_tag(gateway: Gateway, tmp_path: Path) -> None:
    tag: Final = f"tag-reset-{uuid.uuid4().hex}"
    with (
        owned_proxy(
            gateway,
            tmp_path,
            {"PROXY_BUDGET_RESCHEDULER_MIN_TIME": "2", "PROXY_BUDGET_RESCHEDULER_MAX_TIME": "3"},
        ) as candidate,
        candidate.scenario() as scenario,
    ):

        def delete_tag() -> None:
            candidate.post("/tag/delete", {"name": tag})

        model: Final = scenario.model(input_cost_per_token=0.01, output_cost_per_token=0.01)
        candidate.post("/tag/new", {"name": tag, "max_budget": 0.0001, "budget_duration": "5s"})
        scenario.cleanups.callback(delete_tag)
        first: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": f"tag spend {tag}"}],
                "metadata": {"tags": [tag]},
            },
        )
        assert first.status_code == 200, first.text

        def rejection() -> int:
            return candidate.request(
                "POST",
                "/v1/chat/completions",
                {
                    "model": model,
                    "messages": [{"role": "user", "content": f"tag reset probe {tag}"}],
                    "metadata": {"tags": [tag]},
                },
            ).status_code

        status: Final = eventually(rejection, lambda code: code != 200, seconds=70)
        assert status in (400, 422, 429), status
        blocked: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": f"tag reset probe {tag}"}],
                "metadata": {"tags": [tag]},
            },
        )
        assert "budget" in blocked.text.lower(), blocked.text
        recovered: Final = eventually(rejection, lambda code: code == 200, seconds=70)
        assert recovered == 200, recovered
