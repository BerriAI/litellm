import base64
from typing import Final
from urllib.parse import parse_qs, urlsplit

import pytest

from litellm.rust_bridge.traces import query, schema_statements
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec

pytestmark = pytest.mark.requires_rust_extension


@pytest.mark.asyncio
async def test_trace_reader_projects_connection_and_parameters(recording_server: RecordingServer) -> None:
    recording_server.enqueue(ResponseSpec(body={"data": [{"trace_id": "trace-1"}]}))
    rows: Final = await query(
        recording_server.base_url + "?database=wrong&user=wrong&password=wrong",
        "trace_test",
        "reader",
        "p@ss/word%",
        "SELECT {trace_id:String} AS trace_id",
        {"trace_id": "trace-1"},
    )
    request: Final = recording_server.requests[0]
    parameters: Final = parse_qs(urlsplit(request.path).query)
    assert rows == [{"trace_id": "trace-1"}]
    assert request.raw_body == b"SELECT {trace_id:String} AS trace_id"
    assert parameters["database"] == ["trace_test"]
    assert parameters["param_trace_id"] == ["trace-1"]
    assert parameters["readonly"] == ["1"]
    assert "user" not in parameters
    assert "password" not in parameters
    assert request.headers["authorization"] == "Basic " + base64.b64encode(b"reader:p@ss/word%").decode()


@pytest.mark.asyncio
async def test_trace_reader_rejects_success_status_with_embedded_error(recording_server: RecordingServer) -> None:
    recording_server.enqueue(ResponseSpec(body={"data": [], "exception": "query failed"}))
    with pytest.raises(RuntimeError, match="invalid or failed JSON"):
        await query(recording_server.base_url, "trace_test", "reader", "password", "SELECT 1", {})


@pytest.mark.parametrize("database,retention", [("db; DROP DATABASE default", 7), ("traces", 0)])
def test_schema_binding_preserves_configuration_validation(database: str, retention: int) -> None:
    with pytest.raises(ValueError, match="database.*retention"):
        schema_statements(database, retention, 14)
