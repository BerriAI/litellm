from typing import Final

from datadog_reader import DdLogsReader
from datadog_reader import _DdAuthHeaders  # pyright: ignore[reportPrivateUsage]  # verifies private auth-header serialization


def test_failure_diagnostics_hide_credentials_without_changing_auth_headers() -> None:
    api_key: Final = "test-datadog-api-secret"
    app_key: Final = "test-datadog-app-secret"
    reader: Final = DdLogsReader(site="datadoghq.com", api_key=api_key, app_key=app_key)
    headers: Final = _DdAuthHeaders(api_key=api_key, app_key=app_key)

    for value in (reader, headers):
        assert api_key not in repr(value)
        assert app_key not in repr(value)

    assert headers.model_dump(by_alias=True) == {
        "DD-API-KEY": api_key,
        "DD-APPLICATION-KEY": app_key,
    }
