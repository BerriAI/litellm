import pytest
from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration.translation.case import TranslationTestCase
from integration.translation.decisions.bases.openai import GPT_6_LUNA_TEST_CASE
from integration.translation.runner import assert_translation


@pytest.mark.parametrize("case", [GPT_6_LUNA_TEST_CASE], ids=lambda case: case.id)
def test_decisions_basic_openai(case: TranslationTestCase, gateway: Gateway, provider: SharedProvider) -> None:
    assert_translation(case, gateway, provider)
