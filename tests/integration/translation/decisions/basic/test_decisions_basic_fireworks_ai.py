import pytest
from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration.translation.case import TranslationTestCase
from integration.translation.decisions.bases.fireworks_ai import RUNE_SYSTEMONE_TEST_CASE, RUNE_TEST_CASE
from integration.translation.runner import assert_translation


@pytest.mark.parametrize("case", [RUNE_TEST_CASE, RUNE_SYSTEMONE_TEST_CASE], ids=lambda case: case.id)
def test_decisions_basic_fireworks_ai(case: TranslationTestCase, gateway: Gateway, provider: SharedProvider) -> None:
    assert_translation(case, gateway, provider)
