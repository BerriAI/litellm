import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Final

import pytest
from pydantic import PydanticSchemaGenerationError, TypeAdapter

from litellm.proxy.common_utils.openapi_schema_compat import get_openapi_schema_with_compat

UNBLOCK_TIMEOUT_SECONDS: Final = 5


@dataclass(frozen=True, slots=True)
class _BlockingBuild:
    started: threading.Event = field(default_factory=threading.Event)
    may_finish: threading.Event = field(default_factory=threading.Event)

    def __call__(self, **_: object) -> dict[str, object]:
        self.started.set()
        _ = self.may_finish.wait(UNBLOCK_TIMEOUT_SECONDS)
        return {}


def test_only_the_build_documents_a_type_pydantic_cannot_describe():
    class Opaque:
        pass

    def build(**_: object) -> dict[str, object]:
        return {"opaque": TypeAdapter(Opaque).json_schema()}

    schema: Final = get_openapi_schema_with_compat(build, "title", "1.0.0", "", [])

    with pytest.raises(PydanticSchemaGenerationError):
        _ = TypeAdapter(Opaque)
    assert schema == {"opaque": {}}


def test_code_outside_overlapping_builds_still_rejects_unknown_types():
    class Opaque:
        pass

    first: Final = _BlockingBuild()
    second: Final = _BlockingBuild()

    with ThreadPoolExecutor(max_workers=2) as pool:
        first_result: Final = pool.submit(get_openapi_schema_with_compat, first, "title", "1.0.0", "", [])
        _ = first.started.wait(UNBLOCK_TIMEOUT_SECONDS)
        second_result: Final = pool.submit(get_openapi_schema_with_compat, second, "title", "1.0.0", "", [])
        _ = second.started.wait(UNBLOCK_TIMEOUT_SECONDS)
        with pytest.raises(PydanticSchemaGenerationError):
            _ = TypeAdapter(Opaque)
        first.may_finish.set()
        _ = first_result.result()
        second.may_finish.set()
        _ = second_result.result()

    with pytest.raises(PydanticSchemaGenerationError):
        _ = TypeAdapter(Opaque)
