from collections.abc import Generator
from datetime import datetime, timezone
from typing import Final

import pytest

import litellm
from litellm._internal_context import pinned_billing_time
from litellm.types.utils import ModelResponse, PromptTokensDetailsWrapper, Usage


@pytest.fixture
def local_model_cost_map(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
    litellm.get_model_info.cache_clear()
    yield
    litellm.get_model_info.cache_clear()


PEAK_MOMENTS: Final = (
    pytest.param(datetime(2026, 9, 22, 8, 0, tzinfo=timezone.utc), id="tuesday-08:00"),
    pytest.param(datetime(2026, 9, 25, 9, 59, tzinfo=timezone.utc), id="friday-09:59"),
    pytest.param(datetime(2026, 9, 21, 1, 0, tzinfo=timezone.utc), id="monday-01:00"),
)
OFF_PEAK_MOMENTS: Final = (
    pytest.param(datetime(2026, 9, 26, 2, 0, tzinfo=timezone.utc), id="saturday-02:00"),
    pytest.param(datetime(2026, 9, 27, 8, 0, tzinfo=timezone.utc), id="sunday-08:00"),
    pytest.param(datetime(2026, 9, 21, 0, 30, tzinfo=timezone.utc), id="monday-00:30"),
    pytest.param(datetime(2026, 9, 23, 5, 0, tzinfo=timezone.utc), id="wednesday-05:00"),
    pytest.param(datetime(2026, 9, 24, 10, 0, tzinfo=timezone.utc), id="thursday-10:00"),
    pytest.param(datetime(2026, 9, 22, 12, 0, tzinfo=timezone.utc), id="tuesday-12:00"),
)
PEAK_COST_PER_MILLION_IN_AND_OUT_WITH_400K_CACHE_HITS: Final = {
    "deepseek-flash": 1.3824,
    "deepseek-v4-pro": 4.7696,
}


def one_million_in_and_out_with_400k_cache_hits(model: str) -> ModelResponse:
    return ModelResponse(
        model=model,
        usage=Usage(
            prompt_tokens=1_000_000,
            completion_tokens=1_000_000,
            total_tokens=2_000_000,
            prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=400_000),
        ),
    )


def deepseek_cost_at(model: str, moment: datetime) -> float:
    with pinned_billing_time(moment):
        return litellm.completion_cost(
            completion_response=one_million_in_and_out_with_400k_cache_hits(model),
            model=model,
            custom_llm_provider="deepseek",
        )


@pytest.mark.usefixtures("local_model_cost_map")
@pytest.mark.parametrize(("model", "peak_cost"), PEAK_COST_PER_MILLION_IN_AND_OUT_WITH_400K_CACHE_HITS.items())
@pytest.mark.parametrize("moment", PEAK_MOMENTS)
def test_deepseek_bills_the_listed_rate_during_weekday_peak_hours(model: str, peak_cost: float, moment: datetime):
    assert deepseek_cost_at(model, moment) == pytest.approx(peak_cost)


@pytest.mark.usefixtures("local_model_cost_map")
@pytest.mark.parametrize(("model", "peak_cost"), PEAK_COST_PER_MILLION_IN_AND_OUT_WITH_400K_CACHE_HITS.items())
@pytest.mark.parametrize("moment", OFF_PEAK_MOMENTS)
def test_deepseek_bills_half_the_listed_rate_off_peak(model: str, peak_cost: float, moment: datetime):
    assert deepseek_cost_at(model, moment) == pytest.approx(peak_cost / 2)


@pytest.mark.usefixtures("local_model_cost_map")
@pytest.mark.parametrize("alias", ("deepseek-v4-flash", "deepseek-v4-flash-vision-exp", "deepseek/deepseek-flash"))
def test_deepseek_flash_aliases_follow_the_same_off_peak_schedule(alias: str):
    saturday: Final = datetime(2026, 9, 26, 2, 0, tzinfo=timezone.utc)
    assert deepseek_cost_at(alias, saturday) == pytest.approx(
        PEAK_COST_PER_MILLION_IN_AND_OUT_WITH_400K_CACHE_HITS["deepseek-flash"] / 2
    )
