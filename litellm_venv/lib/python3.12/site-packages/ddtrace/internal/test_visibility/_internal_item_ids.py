import dataclasses

from ddtrace.ext.test_visibility import api as ext_api


@dataclasses.dataclass(frozen=True)
class InternalTestId(ext_api.TestId):
    pass
