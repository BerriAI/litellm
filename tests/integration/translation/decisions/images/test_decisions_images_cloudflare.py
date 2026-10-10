import pytest
from integration._support.client import Gateway
from integration._support.provider import SharedProvider
from integration.translation.case import TranslationTestCase
from integration.translation.decisions.bases.cloudflare import (
    CLEF_IMAGES_SYSTEMONE_TEST_CASE,
    CLEF_IMAGES_TEST_CASE,
)
from integration.translation.runner import assert_translation


@pytest.mark.parametrize("case", [CLEF_IMAGES_TEST_CASE, CLEF_IMAGES_SYSTEMONE_TEST_CASE], ids=lambda case: case.id)
def test_decisions_images_cloudflare(case: TranslationTestCase, gateway: Gateway, provider: SharedProvider) -> None:
    assert_translation(case, gateway, provider)
