import uuid
from pathlib import Path
from typing import Final

from integration._support.client import Gateway, eventually
from integration._support.process import owned_proxy


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
