from pathlib import Path
from typing import Final

import pytest

from litellm.rust_bridge import _native  # noqa: F401
from tests.integration._support.native.clickhouse import clickhouse_service as clickhouse_service
from tests.integration._support.native.clickhouse import clickhouse_url as clickhouse_url
from tests.integration._support.native.fixtures import (
    isolate_ocr_test_state as isolate_ocr_test_state,
)
from tests.integration._support.native.fixtures import (
    isolated_azure_auth as isolated_azure_auth,
)
from tests.integration._support.native.fixtures import (
    local_model_cost_map as local_model_cost_map,
)
from tests.integration._support.native.fixtures import (
    recording_server as recording_server,
)
from tests.integration._support.native.fixtures import (
    redis_url as redis_url,
)

_NATIVE_DIR: Final = Path(__file__).resolve().parent


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if _NATIVE_DIR in Path(str(item.fspath)).resolve().parents:
            item.add_marker(pytest.mark.requires_rust_extension)
