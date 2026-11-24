import json
import urllib.parse
import uuid
from collections.abc import Mapping
from itertools import count
from pathlib import Path
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server
from integration.providers.test_bedrock_batch_blank_s3_env_wire import (
    BUCKET,
    JOB_ARN_PREFIX,
    MODEL_ID,
    REGION,
    ROLE_ARN,
    _tls_context,
    bedrock_tunnel,
    s3_peer,
)

LIFECYCLE: Final = ("Submitted", "Validating", "Scheduled", "InProgress")
EXPECTED: Final = ("validating", "validating", "in_progress", "in_progress")


def create_peer(job_arn: str, request: Request) -> Reply:
    if request.method == "POST" and request.target == "/model-invocation-job":
        return Reply(body=json.dumps({"jobArn": job_arn}).encode())
    return Reply(status=404, body=b'{"message": "not scripted"}')


def get_peer(calls: count, job: Mapping[str, object]) -> Reply:
    index: Final = min(next(calls), len(LIFECYCLE) - 1)
    body: Final = {
        **job,
        "status": LIFECYCLE[index],
        "submitTime": "2026-10-01T00:00:00Z",
        "lastModifiedTime": "2026-10-01T00:01:00Z",
    }
    return Reply(body=json.dumps(body).encode())


@pytest.mark.timeout(180)
def test_bedrock_batch_retrieve_reports_lifecycle_status_in_order(gateway: Gateway, tmp_path: Path) -> None:
    job_arn: Final = JOB_ARN_PREFIX + uuid.uuid4().hex
    job: Final = {
        "jobArn": job_arn,
        "modelId": MODEL_ID,
        "inputDataConfig": {"s3InputDataConfig": {"s3Uri": f"s3://{BUCKET}/input.jsonl"}},
        "outputDataConfig": {"s3OutputDataConfig": {"s3Uri": f"s3://{BUCKET}/out/"}},
    }
    calls: Final = count()
    environment: Final = {
        "SSL_VERIFY": "False",
        "AWS_EC2_METADATA_DISABLED": "true",
        "HTTP_PROXY": "",
        "NO_PROXY": "127.0.0.1,localhost",
    }
    with (
        wire_server(s3_peer) as s3,
        wire_server(lambda request: create_peer(job_arn, request), tls=_tls_context(tmp_path)) as bedrock,
        wire_server(lambda request: get_peer(calls, job)) as jobs,
        bedrock_tunnel(bedrock) as tunnel,
        owned_proxy(
            gateway,
            tmp_path,
            {**environment, "HTTPS_PROXY": tunnel.url, "AWS_ENDPOINT_URL_BEDROCK": jobs.url},
        ) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(
            model=f"bedrock/{MODEL_ID}",
            api_key=None,
            api_base=None,
            aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
            aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            aws_region_name=REGION,
            s3_bucket_name=BUCKET,
            s3_endpoint_url=s3.url,
            aws_batch_role_arn=ROLE_ARN,
        )
        line: Final = {
            "custom_id": "req-1",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 8},
        }
        uploaded: Final = candidate.request_multipart(
            "/v1/files",
            {"purpose": "batch", "target_model_names": model},
            {"file": ("in.jsonl", (json.dumps(line) + "\n").encode(), "application/jsonl")},
        )
        assert uploaded.status_code == 200, uploaded.text
        created: Final = candidate.request(
            "POST",
            "/v1/batches",
            {"input_file_id": uploaded.json()["id"], "endpoint": "/v1/chat/completions", "completion_window": "24h"},
        )
        assert created.status_code == 200, created.text
        batch_id: Final = created.json()["id"]

        observed: Final = tuple(candidate.get(f"/v1/batches/{batch_id}")["status"] for _ in range(len(LIFECYCLE)))
        assert observed == EXPECTED, observed

        job_id: Final = job_arn.rsplit("/", 1)[-1]
        drained: Final = jobs.drain()
        gets: Final = tuple(
            request for request in drained if request.method == "GET" and job_id in urllib.parse.unquote(request.target)
        )
        assert len(gets) == len(LIFECYCLE), [(r.method, r.target) for r in drained]
