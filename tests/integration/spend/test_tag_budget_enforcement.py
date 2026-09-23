import uuid
from typing import Final

from integration._support.client import Gateway, eventually


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
