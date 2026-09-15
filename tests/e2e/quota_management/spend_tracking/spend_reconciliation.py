from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from math import isclose
from typing import Final

from e2e_config import provider_edge_base, unique_marker
from e2e_http import unwrap
from lifecycle import ResourceManager
from models import ChatBody, ChatMessage, ChatResponse, KeyGenerateBody, LiteLLMParamsBody, TeamNewBody
from spend_e2e_client import SpendClient

INPUT_RATE: Final = 0.00004
OUTPUT_RATE: Final = 0.00008


@dataclass(frozen=True)
class TeamTraffic:
    team_id: str
    key: str
    responses: tuple[ChatResponse, ...]

    @property
    def prompt_tokens(self) -> int:
        return sum(response.usage.prompt_tokens or 0 for response in self.responses if response.usage)

    @property
    def completion_tokens(self) -> int:
        return sum(response.usage.completion_tokens or 0 for response in self.responses if response.usage)

    @property
    def spend(self) -> float:
        return self.prompt_tokens * INPUT_RATE + self.completion_tokens * OUTPUT_RATE


def create_traffic(client: SpendClient, resources: ResourceManager) -> tuple[TeamTraffic, ...]:
    base: Final = provider_edge_base("openai")
    model: Final = f"e2e-reconciliation-{unique_marker()}"
    model_id: Final = client.proxy.create_model(
        model,
        LiteLLMParamsBody(
            model="openai/gpt-5.6-luna",
            api_key="os.environ/OPENAI_API_KEY",
            api_base=None if base is None else f"{base}/v1",
            input_cost_per_token=INPUT_RATE,
            output_cost_per_token=OUTPUT_RATE,
        ),
    )
    resources.defer(lambda: client.proxy.delete_model(model_id))

    def team_traffic() -> TeamTraffic:
        team: Final = client.proxy.create_team(TeamNewBody(team_alias=f"e2e-spend-{unique_marker()}"))
        resources.defer(lambda: client.proxy.delete_team(team))
        key: Final = client.proxy.generate_key(KeyGenerateBody(team_id=team, models=[model]))
        resources.defer(lambda: client.proxy.delete_key(key))

        prompts: Final = tuple(f"Reply with one word. {index} {unique_marker()}" for index in range(7))

        def call(index: int) -> ChatResponse:
            response: Final = unwrap(
                client.proxy.chat(
                    key,
                    ChatBody(
                        model=model,
                        messages=[ChatMessage(role="user", content=prompts[index])],
                        max_completion_tokens=128,
                    ),
                )
            )
            assert response.id, "successful response must have an ID"
            assert response.usage is not None, "successful response must have usage"
            assert response.usage.prompt_tokens is not None and response.usage.prompt_tokens > 0
            assert response.usage.completion_tokens is not None and response.usage.completion_tokens > 0
            assert response.usage.total_tokens == response.usage.prompt_tokens + response.usage.completion_tokens
            assert not response.usage.cache_creation_input_tokens
            assert not response.usage.cache_read_input_tokens
            assert not response.usage.prompt_tokens_details or not response.usage.prompt_tokens_details.cached_tokens
            return response

        sequential: Final = call(0)
        with ThreadPoolExecutor(max_workers=6) as pool:
            concurrent: Final = tuple(pool.map(call, range(1, 7)))
        return TeamTraffic(team, key, (sequential, *concurrent))

    return tuple(team_traffic() for _ in range(2))


def assert_logs_match(client: SpendClient, traffic: TeamTraffic) -> None:
    expected_ids: Final = frozenset(response.id for response in traffic.responses)
    assert len(expected_ids) == len(traffic.responses), "responses must have distinct IDs"
    rows: Final = client.poll_logs_for_key(
        traffic.key,
        min_rows=len(traffic.responses),
        predicate=lambda values: frozenset(row.request_id for row in values) == expected_ids,
    )
    assert frozenset(row.request_id for row in rows) == expected_ids, "stored IDs must equal returned response IDs"
    assert len(rows) == len(traffic.responses), "expected exactly one scoped spend row per response"
    by_id: Final = {row.request_id: row for row in rows}
    for response in traffic.responses:
        row = by_id[response.id]
        usage = response.usage
        assert usage is not None and usage.prompt_tokens is not None and usage.completion_tokens is not None
        assert row.team_id == traffic.team_id
        assert row.status == "success"
        assert row.cache_hit != "True"
        assert row.prompt_tokens == usage.prompt_tokens
        assert row.completion_tokens == usage.completion_tokens
        assert row.total_tokens == usage.total_tokens
        expected_cost = usage.prompt_tokens * INPUT_RATE + usage.completion_tokens * OUTPUT_RATE
        assert row.spend is not None and isclose(row.spend, expected_cost, rel_tol=1e-6, abs_tol=1e-9)
