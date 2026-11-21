from typing import Final

import pytest

from litellm.litellm_core_utils.get_provider_specific_headers import ProviderSpecificHeaderUtils
from litellm.types.utils import ProviderSpecificHeader

CONFIGURED_HEADERS: Final = {f"x-configured-{index}": "deployment" for index in range(2_000)}
CONDITIONAL_ENTRIES: Final[tuple[ProviderSpecificHeader, ...]] = tuple(
    {
        "custom_llm_provider": "anthropic",
        "extra_headers": {f"x-caller-{index}": "client"},
        "configured_credential_headers": ("x-absent", "X-Configured-1999"),
        "configured_credential_prefix": "deploy",
        "yields_to_configured": True,
    }
    for index in range(2_000)
)


@pytest.mark.benchmark
def test_conditional_entries_beside_many_configured_headers() -> None:
    headers: Final = ProviderSpecificHeaderUtils.get_provider_specific_headers(
        CONDITIONAL_ENTRIES, "anthropic", configured_headers=CONFIGURED_HEADERS
    )
    assert len(headers) == 2_000
