import json
import uuid
from typing import Final

from integration._support.client import Gateway, eventually, object_value
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server

_MODEL: Final = "bedrock/converse/anthropic.claude-sonnet-4-5-20250929-v1:0"
_TOKEN: Final = "synthetic-bedrock-bearer"


def test_bedrock_500_keeps_amzn_request_id_on_error_headers_and_failure_log(gateway: Gateway) -> None:
    identity: Final = f"bedrock-request-id-{uuid.uuid4().hex}"
    amzn_request_id: Final = str(uuid.uuid4())
    prompt: Final = f"failure probe {identity}"

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == "/model/anthropic.claude-sonnet-4-5-20250929-v1%3A0/converse", request.target
        return Reply(
            status=500,
            headers={"x-amzn-RequestId": amzn_request_id},
            body=b'{"message":"synthetic bedrock failure"}',
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=_MODEL,
            api_key=_TOKEN,
            aws_region_name="us-east-1",
            aws_bedrock_runtime_endpoint=wire.url,
            num_retries=0,
        )
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": prompt}]},
        )
        assert response.status_code >= 400, response.text
        assert response.headers.get("llm_provider-x-amzn-requestid") == amzn_request_id, dict(response.headers)
        call_id: Final = response.headers["x-litellm-call-id"]
        assert len(wire.drain()) == 1
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT status, metadata FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (call_id,)
            ),
            lambda values: len(values) == 1,
            seconds=70,
        )
        row: Final = rows[0]
        assert row["status"] == "failure", row
        metadata: Final = row["metadata"]
        parsed: Final = json.loads(metadata) if isinstance(metadata, str) else object_value(metadata)
        error_information: Final = object_value(parsed["error_information"])
        assert error_information["error_provider_request_id"] == amzn_request_id, error_information
