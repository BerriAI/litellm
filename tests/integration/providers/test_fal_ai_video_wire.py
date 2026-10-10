import datetime
import ipaddress
import json
import ssl
import uuid
from pathlib import Path
from typing import Final

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from integration._support.client import JSON_OBJECT, Gateway, object_value, string_value
from integration._support.wire import Reply, Request, wire_server

_MODEL: Final = "bytedance/seedance-2.5/text-to-video"
_H3_MODEL: Final = "minimax/h3/text-to-video"
_MP4: Final = b"\x00\x00\x00\x18ftypmp42" + uuid.uuid4().bytes * 4
_OVERSIZED_SIDE: Final = "9" * 30


def _h3_queue_reply(request_id: str) -> Reply:
    return Reply(body=json.dumps({"status": "IN_QUEUE", "request_id": request_id, "queue_position": 0}).encode())


def _write_self_signed_cert(cert_dir: Path) -> tuple[Path, Path]:
    key: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    now: Final = datetime.datetime.now(datetime.timezone.utc)
    name: Final = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    cert: Final = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=7))
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    cert_file: Final = cert_dir / "cert.pem"
    key_file: Final = cert_dir / "key.pem"
    cert_file.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_file.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )
    return cert_file, key_file


@pytest.mark.covers("other.provider_wire.fal_ai.video_queue_create_status_and_content_download")
def test_fal_video_create_status_and_content_follow_queue_wire_contract(gateway: Gateway) -> None:
    request_id: Final = "fal-req-" + uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        if request.target == f"/files/{request_id}.mp4":
            assert request.method == "GET"
            return Reply(body=_MP4, content_type="video/mp4")
        assert request.headers["authorization"] == "Key synthetic-fal-key"
        if request.method == "POST":
            assert request.target == f"/{_MODEL}"
            assert json.loads(request.body) == {
                "prompt": "a cat playing volleyball on a beach",
                "duration": "4",
                "resolution": "720p",
                "aspect_ratio": "16:9",
            }
            return Reply(
                body=json.dumps({"status": "IN_QUEUE", "request_id": request_id, "queue_position": 0}).encode()
            )
        assert request.method == "GET"
        if request.target == f"/bytedance/seedance-2.5/requests/{request_id}/status":
            return Reply(body=json.dumps({"status": "COMPLETED", "request_id": request_id}).encode())
        assert request.target == f"/bytedance/seedance-2.5/requests/{request_id}"
        return Reply(body=json.dumps({"video": {"url": f"{wire_url}/files/{request_id}.mp4"}}).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        wire_url: Final = wire.url
        model: Final = scenario.model(
            model=f"fal_ai/{_MODEL}",
            api_base=wire.url,
            api_key="synthetic-fal-key",
        )
        created: Final = gateway.post(
            "/v1/videos",
            {
                "model": model,
                "prompt": "a cat playing volleyball on a beach",
                "seconds": "4",
                "size": "1280x720",
            },
        )
        assert created["status"] == "queued"
        video_id: Final = created["id"]
        assert isinstance(video_id, str) and video_id
        status: Final = gateway.get(f"/v1/videos/{video_id}")
        assert status["status"] == "completed"
        content: Final = gateway.request("GET", f"/v1/videos/{video_id}/content")
        assert content.status_code == 200, content.text
        assert content.headers["content-type"].startswith("video/mp4")
        assert content.content == _MP4
        assert [(request.method, request.target) for request in wire.drain()] == [
            ("POST", f"/{_MODEL}"),
            ("GET", f"/bytedance/seedance-2.5/requests/{request_id}/status"),
            ("GET", f"/bytedance/seedance-2.5/requests/{request_id}"),
            ("GET", f"/bytedance/seedance-2.5/requests/{request_id}"),
            ("GET", f"/files/{request_id}.mp4"),
        ]


@pytest.mark.covers("other.provider_wire.fal_ai.video_queue_create_status_and_content_download")
def test_fal_h3_video_create_uses_canonical_body_and_status_path(gateway: Gateway) -> None:
    request_id: Final = "fal-h3-req-" + uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        if request.target == f"/files/{request_id}.mp4":
            assert request.method == "GET"
            return Reply(body=_MP4, content_type="video/mp4")
        assert request.headers["authorization"] == "Key synthetic-fal-key"
        if request.method == "POST":
            assert request.target == f"/{_H3_MODEL}"
            assert json.loads(request.body) == {
                "prompt": "a cat playing volleyball on a beach",
                "duration": 6,
                "resolution": "2K",
            }
            return Reply(
                body=json.dumps({"status": "IN_QUEUE", "request_id": request_id, "queue_position": 0}).encode()
            )
        assert request.method == "GET"
        if request.target == f"/minimax/h3/requests/{request_id}/status":
            return Reply(body=json.dumps({"status": "COMPLETED", "request_id": request_id}).encode())
        assert request.target == f"/minimax/h3/requests/{request_id}"
        return Reply(body=json.dumps({"video": {"url": f"{wire_url}/files/{request_id}.mp4"}}).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        wire_url: Final = wire.url
        model: Final = scenario.model(
            model=f"fal_ai/{_H3_MODEL}",
            api_base=wire.url,
            api_key="synthetic-fal-key",
        )
        created: Final = gateway.post(
            "/v1/videos",
            {
                "model": model,
                "prompt": "a cat playing volleyball on a beach",
                "seconds": 6,
                "size": "2k",
            },
        )
        assert created["status"] == "queued"
        video_id: Final = created["id"]
        status: Final = gateway.get(f"/v1/videos/{video_id}")
        assert status["status"] == "completed"
        content: Final = gateway.request("GET", f"/v1/videos/{video_id}/content")
        assert content.status_code == 200, content.text
        assert content.content == _MP4
        assert [(request.method, request.target) for request in wire.drain()] == [
            ("POST", f"/{_H3_MODEL}"),
            ("GET", f"/minimax/h3/requests/{request_id}/status"),
            ("GET", f"/minimax/h3/requests/{request_id}"),
            ("GET", f"/minimax/h3/requests/{request_id}"),
            ("GET", f"/files/{request_id}.mp4"),
        ]


@pytest.mark.covers("other.provider_wire.fal_ai.video_failed_result_surfaces_fal_error")
def test_fal_video_failed_result_reports_failed_status_and_fal_error(gateway: Gateway) -> None:
    request_id: Final = "fal-failed-req-" + uuid.uuid4().hex
    error_body: Final = {
        "detail": [
            {
                "loc": ["body", "input.reference_image_urls"],
                "msg": "Failed to download the file. Please check if the URL is accessible and try again.",
                "type": "file_download_error",
            }
        ]
    }

    def respond(request: Request) -> Reply:
        assert request.headers["authorization"] == "Key synthetic-fal-key"
        if request.method == "POST":
            assert request.target == f"/{_MODEL}"
            return Reply(
                body=json.dumps({"status": "IN_QUEUE", "request_id": request_id, "queue_position": 0}).encode()
            )
        assert request.method == "GET"
        if request.target == f"/bytedance/seedance-2.5/requests/{request_id}/status":
            return Reply(body=json.dumps({"status": "COMPLETED", "request_id": request_id}).encode())
        assert request.target == f"/bytedance/seedance-2.5/requests/{request_id}"
        return Reply(status=422, body=json.dumps(error_body).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"fal_ai/{_MODEL}",
            api_base=wire.url,
            api_key="synthetic-fal-key",
        )
        created: Final = gateway.post(
            "/v1/videos",
            {
                "model": model,
                "prompt": "a cat playing volleyball on a beach",
                "seconds": "4",
                "size": "1280x720",
            },
        )
        assert created["status"] == "queued"
        video_id: Final = created["id"]
        status: Final = gateway.get(f"/v1/videos/{video_id}")
        assert status["status"] == "failed"
        assert "input.reference_image_urls: Failed to download the file" in string_value(
            object_value(status["error"])["message"]
        )
        content: Final = gateway.request("GET", f"/v1/videos/{video_id}/content")
        assert content.status_code == 422, content.text
        assert "Failed to download the file" in content.text


@pytest.mark.covers("other.provider_wire.fal_ai.video_result_probe_forwards_extra_headers")
def test_fal_result_probe_carries_the_deployment_extra_headers(gateway: Gateway) -> None:
    request_id: Final = "fal-probe-req-" + uuid.uuid4().hex
    marker: Final = uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        assert request.headers["authorization"] == "Key synthetic-fal-key"
        if request.method == "POST":
            assert request.target == f"/{_H3_MODEL}"
            return Reply(
                body=json.dumps({"status": "IN_QUEUE", "request_id": request_id, "queue_position": 0}).encode()
            )
        assert request.method == "GET"
        if request.target == f"/minimax/h3/requests/{request_id}/status":
            return Reply(body=json.dumps({"status": "COMPLETED", "request_id": request_id}).encode())
        assert request.target == f"/minimax/h3/requests/{request_id}"
        if request.headers.get("x-integration-routing") != marker:
            return Reply(status=403, body=json.dumps({"detail": "routing header missing"}).encode())
        return Reply(body=json.dumps({"video": {"url": f"{wire_url}/files/{request_id}.mp4"}}).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        wire_url: Final = wire.url
        model: Final = scenario.model(
            model=f"fal_ai/{_H3_MODEL}",
            api_base=wire.url,
            api_key="synthetic-fal-key",
            extra_headers={"x-integration-routing": marker},
        )
        created: Final = gateway.post(
            "/v1/videos",
            {
                "model": model,
                "prompt": "a paper boat drifting across a puddle after rain",
                "seconds": 6,
                "size": "2k",
            },
        )
        assert created["status"] == "queued"
        video_id: Final = created["id"]
        response: Final = gateway.request("GET", f"/v1/videos/{video_id}")
        assert response.status_code == 200, response.text
        status: Final = JSON_OBJECT.validate_json(response.content)
        assert status["status"] == "completed", status
        assert status["error"] is None, status
        assert [
            (request.method, request.target, request.headers.get("x-integration-routing")) for request in wire.drain()
        ] == [
            ("POST", f"/{_H3_MODEL}", marker),
            ("GET", f"/minimax/h3/requests/{request_id}/status", marker),
            ("GET", f"/minimax/h3/requests/{request_id}", marker),
        ]


@pytest.mark.covers("other.provider_wire.fal_ai.video_result_probe_honors_ssl_verify")
def test_fal_result_probe_reuses_the_ssl_verify_false_client(gateway: Gateway, tmp_path: Path) -> None:
    request_id: Final = "fal-tls-req-" + uuid.uuid4().hex
    cert_file, key_file = _write_self_signed_cert(tmp_path)
    context: Final = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile=cert_file, keyfile=key_file)

    def respond(request: Request) -> Reply:
        assert request.headers["authorization"] == "Key synthetic-fal-key"
        if request.method == "POST":
            assert request.target == f"/{_H3_MODEL}"
            return Reply(
                body=json.dumps({"status": "IN_QUEUE", "request_id": request_id, "queue_position": 0}).encode()
            )
        assert request.method == "GET"
        if request.target == f"/minimax/h3/requests/{request_id}/status":
            return Reply(body=json.dumps({"status": "COMPLETED", "request_id": request_id}).encode())
        assert request.target == f"/minimax/h3/requests/{request_id}"
        return Reply(body=json.dumps({"video": {"url": f"{wire_url}/files/{request_id}.mp4"}}).encode())

    with wire_server(respond, tls=context) as wire, gateway.scenario() as scenario:
        wire_url: Final = wire.url
        model: Final = scenario.model(
            model=f"fal_ai/{_H3_MODEL}",
            api_base=wire.url,
            api_key="synthetic-fal-key",
            ssl_verify=False,
        )
        created: Final = gateway.post(
            "/v1/videos",
            {
                "model": model,
                "prompt": "a paper boat drifting across a puddle after rain",
                "seconds": 6,
                "size": "2k",
            },
        )
        assert created["status"] == "queued"
        video_id: Final = created["id"]
        response: Final = gateway.request("GET", f"/v1/videos/{video_id}")
        assert response.status_code == 200, response.text
        status: Final = JSON_OBJECT.validate_json(response.content)
        assert status["status"] == "completed", status
        assert status["error"] is None, status
        assert [(request.method, request.target) for request in wire.drain()] == [
            ("POST", f"/{_H3_MODEL}"),
            ("GET", f"/minimax/h3/requests/{request_id}/status"),
            ("GET", f"/minimax/h3/requests/{request_id}"),
        ]


@pytest.mark.covers("other.provider_wire.fal_ai.video_result_probe_hangup_stays_completed")
def test_fal_provider_hanging_up_on_the_result_probe_keeps_the_completed_status(gateway: Gateway) -> None:
    request_id: Final = "fal-hangup-req-" + uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        assert request.headers["authorization"] == "Key synthetic-fal-key"
        if request.method == "POST":
            assert request.target == f"/{_H3_MODEL}"
            return Reply(
                body=json.dumps({"status": "IN_QUEUE", "request_id": request_id, "queue_position": 0}).encode()
            )
        assert request.method == "GET"
        if request.target == f"/minimax/h3/requests/{request_id}/status":
            return Reply(body=json.dumps({"status": "COMPLETED", "request_id": request_id}).encode())
        assert request.target == f"/minimax/h3/requests/{request_id}"
        return Reply(chunks=(b"{",), abort_after=0)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"fal_ai/{_H3_MODEL}",
            api_base=wire.url,
            api_key="synthetic-fal-key",
        )
        created: Final = gateway.post(
            "/v1/videos",
            {
                "model": model,
                "prompt": "a paper boat drifting across a puddle after rain",
                "seconds": 6,
                "size": "2k",
            },
        )
        assert created["status"] == "queued"
        video_id: Final = created["id"]
        response: Final = gateway.request("GET", f"/v1/videos/{video_id}")
        assert response.status_code == 200, response.text
        status: Final = JSON_OBJECT.validate_json(response.content)
        assert status["status"] == "completed", status
        assert status["error"] is None, status
        assert [(request.method, request.target) for request in wire.drain()] == [
            ("POST", f"/{_H3_MODEL}"),
            ("GET", f"/minimax/h3/requests/{request_id}/status"),
            ("GET", f"/minimax/h3/requests/{request_id}"),
        ]


@pytest.mark.covers("other.provider_wire.fal_ai.h3_auto_duration_omits_duration_and_queues")
def test_fal_h3_auto_duration_omits_duration_and_queues(gateway: Gateway) -> None:
    request_id: Final = "fal-h3-auto-" + uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == f"/{_H3_MODEL}"
        assert json.loads(request.body) == {
            "prompt": "a cat playing volleyball on a beach",
            "resolution": "768P",
            "aspect_ratio": "16:9",
        }
        return _h3_queue_reply(request_id)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"fal_ai/{_H3_MODEL}", api_base=wire.url, api_key="synthetic-fal-key")
        response: Final = gateway.request(
            "POST",
            "/v1/videos",
            {"model": model, "prompt": "a cat playing volleyball on a beach", "seconds": "auto", "size": "1280x720"},
        )
        assert response.status_code == 200, response.text
        assert response.json()["status"] == "queued"
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", f"/{_H3_MODEL}")]


@pytest.mark.covers("other.provider_wire.fal_ai.h3_oversized_size_uses_top_resolution_tier")
def test_fal_h3_oversized_size_uses_top_resolution_tier_and_queues(gateway: Gateway) -> None:
    request_id: Final = "fal-h3-oversized-" + uuid.uuid4().hex

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == f"/{_H3_MODEL}"
        assert json.loads(request.body) == {
            "prompt": "a cat playing volleyball on a beach",
            "duration": 5,
            "resolution": "4K",
            "aspect_ratio": "1:1",
        }
        return _h3_queue_reply(request_id)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"fal_ai/{_H3_MODEL}", api_base=wire.url, api_key="synthetic-fal-key")
        response: Final = gateway.request(
            "POST",
            "/v1/videos",
            {
                "model": model,
                "prompt": "a cat playing volleyball on a beach",
                "seconds": "5",
                "size": f"{_OVERSIZED_SIDE}x{_OVERSIZED_SIDE}",
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["status"] == "queued"
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", f"/{_H3_MODEL}")]
