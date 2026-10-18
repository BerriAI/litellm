import pytest
from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration.translation.case import TranslationTestCase
from integration.translation.decisions.bases.typesafe import JEV_1_13_0_TEST_CASE, JEV_1_13_0_SYSTEMONE_TEST_CASE
from integration.translation.runner import assert_translation


@pytest.mark.parametrize("case", [JEV_1_13_0_TEST_CASE, JEV_1_13_0_SYSTEMONE_TEST_CASE], ids=lambda case: case.id)
def test_decisions_basic_typesafe(case: TranslationTestCase, gateway: Gateway, provider: SharedProvider) -> None:
    assert_translation(case, gateway, provider)
