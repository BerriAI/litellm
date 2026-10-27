import pytest
from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration.translation.case import TranslationTestCase
from integration.translation.decisions.bases.perplexity import (
    PPLX_DECIDER_V1_27B_TEST_CASE,
    PPLX_DECIDER_V1_27B_SYSTEMONE_TEST_CASE,
)
from integration.translation.runner import assert_translation


@pytest.mark.parametrize(
    "case", [PPLX_DECIDER_V1_27B_TEST_CASE, PPLX_DECIDER_V1_27B_SYSTEMONE_TEST_CASE], ids=lambda case: case.id
)
def test_decisions_basic_perplexity(case: TranslationTestCase, gateway: Gateway, provider: SharedProvider) -> None:
    assert_translation(case, gateway, provider)
