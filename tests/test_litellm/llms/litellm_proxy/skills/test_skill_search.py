import asyncio
import json
from collections.abc import Sequence
from types import MappingProxyType
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai import APIConnectionError

import litellm
from litellm.llms.litellm_proxy.skills.skill_search import (
    MAX_SKILL_SEARCH_TEXT_CHARS,
    SkillSearchEmbeddingFailed,
    SkillSearchHits,
    SkillSearchIndex,
    SkillSearchNotConfigured,
    search_skills,
    skill_search_text,
)
from litellm.proxy._types import LiteLLM_SkillsTable, UserAPIKeyAuth
from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError
from litellm.proxy.common_utils.semantic_text_index import Vector, cosine_similarity

CALLER: Final = UserAPIKeyAuth(api_key="hashed-caller-key", team_id="team-1", user_id="user-1")

TRANSLATOR: Final = LiteLLM_SkillsTable(
    skill_id="translate-file",
    display_title="Document Translator",
    description="Converts files from one language into another",
    instructions="Take an uploaded document and produce it in the target language",
)
SQL_ANALYST: Final = LiteLLM_SkillsTable(
    skill_id="warehouse-sql-analyst",
    display_title="Warehouse SQL Analyst",
    description="Runs SQL against the inventory database",
)
TRIP_PLANNER: Final = LiteLLM_SkillsTable(
    skill_id="trip-planner",
    display_title="Trip Planner",
    description="Books flights and hotels",
)
SKILLS: Final = (TRANSLATOR, SQL_ANALYST, TRIP_PLANNER)

VECTORS: Final = MappingProxyType(
    {
        "language translation": (1.0, 0.0, 0.0),
        skill_search_text(TRANSLATOR): (0.9, 0.1, 0.0),
        skill_search_text(SQL_ANALYST): (0.0, 1.0, 0.0),
        skill_search_text(TRIP_PLANNER): (0.3, 0.0, 1.0),
    }
)


def _pass_through_key_limits() -> MagicMock:
    limits = MagicMock()
    limits.pre_call_hook = AsyncMock(side_effect=lambda user_api_key_dict, data, call_type: data)
    return limits


def _embedding_router() -> MagicMock:
    router = MagicMock()
    router.aembedding = AsyncMock(
        side_effect=lambda model, input, metadata: litellm.EmbeddingResponse(
            model=model,
            data=[{"object": "embedding", "index": i, "embedding": list(VECTORS[t])} for i, t in enumerate(input)],
        )
    )
    return router


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []  # mutable-ok: test spy recording embed inputs

    async def __call__(self, texts: Sequence[str]) -> Sequence[Vector]:
        self.calls.append(tuple(texts))
        return tuple(VECTORS[text] for text in texts)


class FixedDimensionEmbedder:
    def __init__(self, dimensions: int) -> None:
        self.dimensions: Final = dimensions
        self.calls: list[tuple[str, ...]] = []  # mutable-ok: test spy recording embed inputs

    async def __call__(self, texts: Sequence[str]) -> Sequence[Vector]:
        self.calls.append(tuple(texts))
        await asyncio.sleep(0)
        return tuple((1.0,) * self.dimensions for _ in texts)


class TestSkillSearchText:
    def test_joins_title_description_and_instructions(self) -> None:
        assert skill_search_text(TRANSLATOR) == (
            "Document Translator\n"
            "Converts files from one language into another\n"
            "Take an uploaded document and produce it in the target language"
        )

    def test_missing_fields_fall_back_to_whatever_is_present(self) -> None:
        assert skill_search_text(LiteLLM_SkillsTable(skill_id="bare", display_title="bare")) == "bare"

    def test_all_fields_absent_is_an_empty_string(self) -> None:
        assert skill_search_text(LiteLLM_SkillsTable(skill_id="empty")) == ""

    def test_oversized_instructions_are_cut_so_one_skill_cannot_blow_up_the_embedding_batch(self) -> None:
        bloated = LiteLLM_SkillsTable(
            skill_id="bloated", display_title="Bloated", instructions="x" * (MAX_SKILL_SEARCH_TEXT_CHARS * 3)
        )
        text = skill_search_text(bloated)
        assert len(text) == MAX_SKILL_SEARCH_TEXT_CHARS
        assert text.startswith("Bloated\n")


class TestCosineSimilarity:
    def test_identical_direction_scores_one(self) -> None:
        assert cosine_similarity((2.0, 0.0), (1.0, 0.0)) == pytest.approx(1.0)

    def test_orthogonal_scores_zero(self) -> None:
        assert cosine_similarity((1.0, 0.0), (0.0, 1.0)) == pytest.approx(0.0)

    def test_zero_vector_scores_zero_instead_of_dividing(self) -> None:
        assert cosine_similarity((0.0, 0.0), (1.0, 0.0)) == 0.0


class TestSkillSearchIndex:
    @pytest.mark.asyncio
    async def test_ranks_by_similarity_and_truncates_to_top_k(self) -> None:
        outcome = await SkillSearchIndex().search(
            "language translation", SKILLS, top_k=2, embed=FakeEmbedder(), embedding_model="m"
        )
        assert isinstance(outcome, SkillSearchHits)
        assert [hit.skill.skill_id for hit in outcome.hits] == ["translate-file", "trip-planner"]
        assert outcome.hits[0].score > outcome.hits[1].score

    @pytest.mark.asyncio
    async def test_second_search_only_embeds_the_query(self) -> None:
        index = SkillSearchIndex()
        embedder = FakeEmbedder()
        await index.search("language translation", SKILLS, top_k=5, embed=embedder, embedding_model="m")
        await index.search("language translation", SKILLS, top_k=5, embed=embedder, embedding_model="m")
        assert len(embedder.calls[0]) == 1 + len(SKILLS)
        assert embedder.calls[1] == ("language translation",)

    @pytest.mark.asyncio
    async def test_switching_embedding_models_does_not_reuse_cached_vectors(self) -> None:
        index = SkillSearchIndex()
        await index.search("language translation", SKILLS, top_k=5, embed=FakeEmbedder(), embedding_model="small")
        wide = FixedDimensionEmbedder(2)
        outcome = await index.search("language translation", SKILLS, top_k=5, embed=wide, embedding_model="wide")
        assert isinstance(outcome, SkillSearchHits)
        assert len(wide.calls[0]) == 1 + len(SKILLS)

    @pytest.mark.asyncio
    async def test_cached_vectors_of_another_dimension_are_re_embedded(self) -> None:
        index = SkillSearchIndex()
        await index.search("language translation", SKILLS, top_k=5, embed=FakeEmbedder(), embedding_model="m")
        fallback = FixedDimensionEmbedder(2)
        outcome = await index.search("language translation", SKILLS, top_k=5, embed=fallback, embedding_model="m")
        assert isinstance(outcome, SkillSearchHits)
        assert fallback.calls == [
            ("language translation",),
            ("language translation", *(skill_search_text(skill) for skill in SKILLS)),
        ]

    @pytest.mark.asyncio
    async def test_re_embedding_a_subset_drops_the_other_skills_old_vectors(self) -> None:
        index = SkillSearchIndex()
        await index.search("language translation", SKILLS, top_k=5, embed=FakeEmbedder(), embedding_model="m")
        wide = FixedDimensionEmbedder(2)
        await index.search("language translation", SKILLS[:1], top_k=5, embed=wide, embedding_model="m")
        await index.search("language translation", SKILLS, top_k=5, embed=wide, embedding_model="m")
        assert wide.calls[-1] == ("language translation", *(skill_search_text(skill) for skill in SKILLS[1:]))

    @pytest.mark.asyncio
    async def test_concurrent_searches_keep_each_others_vectors(self) -> None:
        index = SkillSearchIndex()
        embedder = FixedDimensionEmbedder(3)
        await asyncio.gather(
            index.search("q", SKILLS[:1], top_k=5, embed=embedder, embedding_model="m"),
            index.search("q", SKILLS[1:], top_k=5, embed=embedder, embedding_model="m"),
        )
        await index.search("q", SKILLS, top_k=5, embed=embedder, embedding_model="m")
        assert embedder.calls[-1] == ("q",)

    @pytest.mark.asyncio
    async def test_least_recently_searched_skills_are_evicted_once_the_index_is_full(self) -> None:
        index = SkillSearchIndex(max_entries=len(SKILLS))
        embedder = FixedDimensionEmbedder(3)
        newcomer = LiteLLM_SkillsTable(skill_id="newcomer", display_title="Newcomer")
        await index.search("q", SKILLS, top_k=5, embed=embedder, embedding_model="m")
        await index.search("q", SKILLS[:1], top_k=5, embed=embedder, embedding_model="m")
        await index.search("q", (newcomer,), top_k=5, embed=embedder, embedding_model="m")
        await index.search("q", SKILLS, top_k=5, embed=embedder, embedding_model="m")
        assert embedder.calls[-1] == ("q", skill_search_text(SKILLS[1]))

    @pytest.mark.asyncio
    async def test_deleted_skills_stop_occupying_the_index_after_enough_new_ones(self) -> None:
        index = SkillSearchIndex(max_entries=2)
        embedder = FixedDimensionEmbedder(3)
        for generation in range(50):
            skill = LiteLLM_SkillsTable(skill_id=f"gen-{generation}", display_title=f"Generation {generation}")
            await index.search("q", (skill,), top_k=5, embed=embedder, embedding_model="m")
        await index.search("q", SKILLS, top_k=5, embed=embedder, embedding_model="m")
        await index.search("q", SKILLS, top_k=5, embed=embedder, embedding_model="m")
        assert embedder.calls[-1] == ("q", skill_search_text(SKILLS[0]))

    @pytest.mark.asyncio
    async def test_mixed_dimensions_in_one_batch_become_embedding_failed(self) -> None:
        async def mixed(texts: Sequence[str]) -> Sequence[Vector]:
            return ((1.0, 0.0), *((1.0, 0.0, 0.0) for _ in texts[1:]))

        outcome = await SkillSearchIndex().search("q", SKILLS, top_k=5, embed=mixed, embedding_model="m")
        assert isinstance(outcome, SkillSearchEmbeddingFailed)
        assert "mixed dimensions" in outcome.reason

    @pytest.mark.asyncio
    async def test_no_accessible_skills_returns_no_hits_without_embedding(self) -> None:
        embedder = FakeEmbedder()
        outcome = await SkillSearchIndex().search("anything", (), top_k=5, embed=embedder, embedding_model="m")
        assert outcome == SkillSearchHits(hits=())
        assert embedder.calls == []

    @pytest.mark.asyncio
    async def test_provider_error_becomes_embedding_failed(self) -> None:
        async def failing(texts: Sequence[str]) -> Sequence[Vector]:
            raise APIConnectionError(request=MagicMock())

        outcome = await SkillSearchIndex().search("q", SKILLS, top_k=5, embed=failing, embedding_model="m")
        assert isinstance(outcome, SkillSearchEmbeddingFailed)
        assert "embedding the search query failed" in outcome.reason

    @pytest.mark.asyncio
    async def test_wrong_vector_count_becomes_embedding_failed(self) -> None:
        async def short(texts: Sequence[str]) -> Sequence[Vector]:
            return ((1.0, 0.0, 0.0),)

        outcome = await SkillSearchIndex().search("q", SKILLS, top_k=5, embed=short, embedding_model="m")
        assert isinstance(outcome, SkillSearchEmbeddingFailed)


class TestSearchSkills:
    @pytest.mark.asyncio
    async def test_no_embedding_model_is_not_configured(self) -> None:
        outcome = await search_skills(
            "q",
            SKILLS,
            5,
            router=MagicMock(),
            embedding_model=None,
            index=SkillSearchIndex(),
            user_api_key_dict=CALLER,
            proxy_logging_obj=_pass_through_key_limits(),
        )
        assert isinstance(outcome, SkillSearchNotConfigured)
        assert "skill_search_embedding_model" in outcome.reason

    @pytest.mark.asyncio
    async def test_no_router_is_not_configured(self) -> None:
        outcome = await search_skills(
            "q",
            SKILLS,
            5,
            router=None,
            embedding_model="m",
            index=SkillSearchIndex(),
            user_api_key_dict=CALLER,
            proxy_logging_obj=_pass_through_key_limits(),
        )
        assert isinstance(outcome, SkillSearchNotConfigured)

    @pytest.mark.asyncio
    async def test_router_embeddings_are_read_from_the_response(self) -> None:
        router = _embedding_router()
        outcome = await search_skills(
            "language translation",
            SKILLS,
            1,
            router=router,
            embedding_model="text-embedding-3-small",
            index=SkillSearchIndex(),
            user_api_key_dict=CALLER,
            proxy_logging_obj=_pass_through_key_limits(),
        )
        assert isinstance(outcome, SkillSearchHits)
        assert [hit.skill.skill_id for hit in outcome.hits] == ["translate-file"]
        assert router.aembedding.await_args.kwargs["model"] == "text-embedding-3-small"

    @pytest.mark.asyncio
    async def test_embedding_spend_is_attributed_to_the_calling_key(self) -> None:
        router = _embedding_router()
        await search_skills(
            "language translation",
            SKILLS,
            1,
            router=router,
            embedding_model="text-embedding-3-small",
            index=SkillSearchIndex(),
            user_api_key_dict=CALLER,
            proxy_logging_obj=_pass_through_key_limits(),
        )
        metadata = router.aembedding.await_args.kwargs["metadata"]
        assert metadata["user_api_key"] == "hashed-caller-key"
        assert metadata["user_api_key_team_id"] == "team-1"
        assert metadata["user_api_key_user_id"] == "user-1"

    @pytest.mark.asyncio
    async def test_key_limits_are_checked_against_the_real_embedding_call_before_it_runs(self) -> None:
        router = _embedding_router()
        key_limits = _pass_through_key_limits()
        await search_skills(
            "language translation",
            SKILLS,
            1,
            router=router,
            embedding_model="text-embedding-3-small",
            index=SkillSearchIndex(),
            user_api_key_dict=CALLER,
            proxy_logging_obj=key_limits,
        )
        checked = key_limits.pre_call_hook.await_args.kwargs
        assert checked["user_api_key_dict"] is CALLER
        assert checked["call_type"] == "aembedding"
        assert checked["data"]["model"] == "text-embedding-3-small"
        assert checked["data"]["input"] == router.aembedding.await_args.kwargs["input"]
        assert checked["data"]["metadata"]["user_api_key"] == "hashed-caller-key"

    @pytest.mark.asyncio
    async def test_the_embedding_model_sees_the_request_as_the_guardrails_rewrote_it(self) -> None:
        router = MagicMock()
        router.aembedding = AsyncMock(
            side_effect=lambda model, input, metadata: litellm.EmbeddingResponse(
                model=model,
                data=[{"object": "embedding", "index": i, "embedding": [1.0, 0.0, 0.0]} for i in range(len(input))],
            )
        )
        key_limits = MagicMock()
        key_limits.pre_call_hook = AsyncMock(
            side_effect=lambda user_api_key_dict, data, call_type: {
                **data,
                "input": ["[MASKED]" for _ in data["input"]],
                "metadata": {**data["metadata"], "guardrail": "masked"},
            }
        )
        await search_skills(
            "language translation",
            SKILLS,
            1,
            router=router,
            embedding_model="text-embedding-3-small",
            index=SkillSearchIndex(),
            user_api_key_dict=CALLER,
            proxy_logging_obj=key_limits,
        )
        sent = tuple(call.kwargs for call in router.aembedding.await_args_list)
        assert sent
        assert all(set(call["input"]) == {"[MASKED]"} for call in sent)
        assert all(call["metadata"]["guardrail"] == "masked" for call in sent)

    @pytest.mark.asyncio
    async def test_a_key_over_its_limit_never_reaches_the_embedding_model(self) -> None:
        router = _embedding_router()
        key_limits = MagicMock()
        key_limits.pre_call_hook = AsyncMock(side_effect=ProxyRateLimitError(detail="rpm exceeded"))
        with pytest.raises(ProxyRateLimitError) as raised:
            await search_skills(
                "language translation",
                SKILLS,
                1,
                router=router,
                embedding_model="text-embedding-3-small",
                index=SkillSearchIndex(),
                user_api_key_dict=CALLER,
                proxy_logging_obj=key_limits,
            )
        assert raised.value.status_code == 429
        router.aembedding.assert_not_awaited()


@pytest.fixture
def accessible_skills(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    list_for_search = AsyncMock(return_value=list(SKILLS))
    monkeypatch.setattr(
        "litellm.llms.litellm_proxy.skills.handler.LiteLLMSkillsHandler.list_skills_for_search", list_for_search
    )
    return list_for_search


@pytest.fixture
def key_limits(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    limits = _pass_through_key_limits()
    monkeypatch.setattr("litellm.proxy.proxy_server.proxy_logging_obj", limits)
    return limits


@pytest.fixture
def embedding_router(monkeypatch: pytest.MonkeyPatch, key_limits: MagicMock) -> MagicMock:
    router = _embedding_router()
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", router)
    monkeypatch.setattr(litellm, "skill_search_embedding_model", "text-embedding-3-small")
    return router


class TestHandleSkillSearchMCP:
    @pytest.mark.asyncio
    async def test_top_k_is_clamped_to_the_same_ceiling_as_rest(
        self, accessible_skills: AsyncMock, embedding_router: MagicMock
    ) -> None:
        from litellm.llms.litellm_proxy.skills.skill_search import MAX_SKILL_SEARCH_TOP_K
        from litellm.proxy._experimental.mcp_server.tool_search import handle_skill_search

        many_skills: Final = tuple(
            LiteLLM_SkillsTable(
                skill_id=f"skill-{i}", display_title=TRIP_PLANNER.display_title, description=TRIP_PLANNER.description
            )
            for i in range(MAX_SKILL_SEARCH_TOP_K + 50)
        )
        accessible_skills.return_value = list(many_skills)

        result = await handle_skill_search(
            query="language translation", top_k=10_000, user_api_key_dict=UserAPIKeyAuth(user_id="u")
        )
        assert result.isError is False
        assert len(json.loads(result.content[0].text)) == MAX_SKILL_SEARCH_TOP_K

    @pytest.mark.asyncio
    async def test_top_k_below_one_is_raised_to_one(
        self, accessible_skills: AsyncMock, embedding_router: MagicMock
    ) -> None:
        from litellm.proxy._experimental.mcp_server.tool_search import handle_skill_search

        result = await handle_skill_search(
            query="language translation", top_k=0, user_api_key_dict=UserAPIKeyAuth(user_id="u")
        )
        assert result.isError is False
        assert len(json.loads(result.content[0].text)) == 1
