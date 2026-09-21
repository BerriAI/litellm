import asyncio
import base64
import hashlib
import hmac
import json
from datetime import datetime
from typing import Final

import httpx
import pytest

from integration._support.sigv4 import encoded_path, signature
from integration._support.wire import Reply, Request, wire_server

ACCESS: Final = "AKIAIOSFODNN7EXAMPLE"
SECRET: Final = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"


@pytest.mark.covers("other.provider_wire.s3.verifier_known_answer_and_negative_controls")
def test_sigv4_verifier_matches_published_put_and_rejects_corruption() -> None:
    # Public AWS example credentials and PUT vector, not an active account:
    # https://docs.aws.amazon.com/AmazonS3/latest/developerguide/sig-v4-header-based-auth.html
    headers: Final = {
        "date": "Fri, 24 May 2013 00:00:00 GMT", "host": "examplebucket.s3.amazonaws.com",
        "x-amz-content-sha256": "44ce7dd67c959e0d3524ffac1771dfbba87d2b6b4b4e99e42034a8b803f8b072",
        "x-amz-date": "20130524T000000Z", "x-amz-storage-class": "REDUCED_REDUNDANCY",
    }
    signed: Final = "date;host;x-amz-content-sha256;x-amz-date;x-amz-storage-class"
    expected: Final = (
        "9e0e90d9c76de8fa5b200d8c849cd5b8dc7a3be3951ddb7f6a76b4158342019d",
        "98ad721746da40c64f1a55b78f14c238d841ea1380cd77a1b5971af0ece108bd",
    )
    actual: Final = signature("PUT", "/test%24file.text", headers, signed, b"Welcome to Amazon S3.", SECRET, "20130524/us-east-1/s3/aws4_request")
    assert actual == expected
    assert signature("PUT", "/test$file.text", headers, signed, b"Welcome to Amazon S3.", SECRET, "20130524/us-east-1/s3/aws4_request") != expected
    assert encoded_path("/bucket/a=b+c/d e/雪.json") == "/bucket/a%3Db%2Bc/d%20e/%E9%9B%AA.json"


@pytest.mark.covers("other.provider_wire.s3.sync_async_reserved_keys_are_signed_and_accepted")
async def test_s3_sync_and_async_uploads_pass_independent_wire_verification(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.integrations.s3_v2 import S3Logger
    from litellm.types.integrations.s3_v2 import s3BatchLoggingElement

    monkeypatch.setattr("botocore.auth.get_current_datetime", lambda: datetime(2026, 9, 14))
    payload: Final = {"id": "synthetic-event", "content": "synthetic snow 雪"}
    expected_path = ""

    def verify(request: Request) -> Reply:
        if request.method != "PUT" or request.target != expected_path:
            return Reply(status=403)
        try:
            authorization: Final = request.headers.get("authorization", "")
            assert authorization.startswith("AWS4-HMAC-SHA256 ")
            fields: Final = dict(part.split("=", 1) for part in authorization.removeprefix("AWS4-HMAC-SHA256 ").split(", "))
            access, scope = fields["Credential"].split("/", 1)
            assert access == ACCESS and scope == "20260914/us-east-1/s3/aws4_request"
            assert request.headers["x-amz-date"] == "20260914T000000Z"
            signed: Final = fields["SignedHeaders"].split(";")
            assert signed == sorted(set(signed))
            assert {"host", "content-md5", "x-amz-date"}.issubset(signed)
            assert {name for name in request.headers if name.startswith("x-amz-") and name != "x-amz-content-sha256"}.issubset(signed)
            assert request.headers["content-md5"] == base64.b64encode(hashlib.md5(request.body, usedforsecurity=False).digest()).decode()
            assert request.headers["x-amz-content-sha256"] == hashlib.sha256(request.body).hexdigest()
            expected: Final = signature("PUT", request.target, request.headers, fields["SignedHeaders"], request.body, SECRET, scope)[1]
            return Reply(status=200 if hmac.compare_digest(expected, fields["Signature"]) else 403)
        except (AssertionError, KeyError, ValueError):
            return Reply(status=403)

    with wire_server(verify) as wire:
        prior: Final = asyncio.all_tasks()
        logger: Final = S3Logger(s3_bucket_name="integration-bucket", s3_region_name="us-east-1", s3_endpoint_url=wire.url,
                                s3_aws_access_key_id=ACCESS, s3_aws_secret_access_key=SECRET, s3_callback_params_override={})
        owned: Final = asyncio.all_tasks() - prior
        assert len(owned) == 1
        try:
            for mode in ("sync", "async"):
                for key in ("plain.json", "a=b+c/d e/雪.json", "percent%2Fplus+.json"):
                    expected_path = encoded_path(f"/integration-bucket/{key}")
                    element: Final = s3BatchLoggingElement(payload=payload, s3_object_key=key, s3_object_download_filename="event.json")
                    if mode == "sync":
                        await asyncio.to_thread(logger.upload_data_to_s3, element)
                    else:
                        await logger.async_upload_data_to_s3(element)
                    requests: Final = wire.drain()
                    assert len(requests) == 1, "Upload must be accepted on its first actual PUT"
                    request: Final = requests[0]
                    assert request.target == expected_path
                    assert json.loads(request.body) == payload
                    assert verify(request).status == 200
                    with httpx.Client(timeout=5, trust_env=False) as client:
                        corrupt: Final = {**request.headers, "authorization": request.headers["authorization"][:-1] + ("0" if request.headers["authorization"][-1] != "0" else "1")}
                        assert client.put(wire.url + expected_path, content=request.body, headers=corrupt).status_code == 403
                        assert client.put(wire.url + expected_path + "-wrong", content=request.body, headers=request.headers).status_code == 403
                        assert client.put(wire.url + expected_path, content=request.body + b" ", headers={name: value for name, value in request.headers.items() if name != "content-length"}).status_code == 403
                        fields: Final = dict(part.split("=", 1) for part in request.headers["authorization"].removeprefix("AWS4-HMAC-SHA256 ").split(", "))
                        for signed, scope, md5 in (
                            (fields["SignedHeaders"].replace("host;", ""), "20260914/us-east-1/s3/aws4_request", request.headers["content-md5"]),
                            (fields["SignedHeaders"], "20260914/us-west-2/s3/aws4_request", request.headers["content-md5"]),
                            (fields["SignedHeaders"], "20260914/us-east-1/s3/aws4_request", "AAAAAAAAAAAAAAAAAAAAAA=="),
                        ):
                            candidate_headers: Final = {**request.headers, "content-md5": md5}
                            digest: Final = signature("PUT", request.target, candidate_headers, signed, request.body, SECRET, scope)[1]
                            candidate_headers["authorization"] = f"AWS4-HMAC-SHA256 Credential={ACCESS}/{scope}, SignedHeaders={signed}, Signature={digest}"
                            assert client.put(wire.url + expected_path, content=request.body, headers=candidate_headers).status_code == 403
                    assert len(wire.drain()) == 6

        finally:
            for task in owned:
                task.cancel()
            await asyncio.gather(*owned, return_exceptions=True)
            assert all(task.done() for task in owned)
