import asyncio
import io
import ipaddress
import json
import ssl
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

import litellm
from litellm.llms.flowspeech.text_to_speech.transformation import FlowSpeechException

REQUEST_ID = "b698824b-de05-486d-8833-05d5c4522a39"
JOB_ID = "11111111-2222-4333-8444-555555555555"
BUDGET = 0.3


def wav_bytes():
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        audio.writeframes(b"\x00\x00" * 24)
    return buffer.getvalue()


def tls_contexts(directory):
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(1)
        .not_valid_before(datetime(2020, 1, 1, tzinfo=timezone.utc))
        .not_valid_after(datetime(2100, 1, 1, tzinfo=timezone.utc))
        .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]), False)
        .sign(key, hashes.SHA256())
    )
    cert_file = directory / "loopback.pem"
    key_file = directory / "loopback.key"
    cert_file.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_file.write_bytes(
        key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    )
    server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_context.load_cert_chain(cert_file, key_file)
    return server_context, ssl.create_default_context(cafile=str(cert_file))


@pytest.fixture
def slow_speech_peer(tmp_path, scenario):
    requests = []
    body = wav_bytes()
    server_context, client_context = tls_contexts(tmp_path)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def setup(self):
            super().setup()
            self.connection.settimeout(2)

        def log_message(self, format, *args):
            return

        def reply(self, content):
            self.send_response(200)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            self.wfile.flush()

        def do_POST(self):
            request_body = self.rfile.read(int(self.headers["Content-Length"]))
            requests.append((self.command, self.path, dict(self.headers), json.loads(request_body)))
            self.reply(
                json.dumps(
                    {
                        "id": JOB_ID,
                        "requestId": self.headers["Idempotency-Key"],
                        "status": "success",
                        "usage": {"billableCharacters": 5, "reservedCredits": 0, "chargedCredits": 5},
                        "result": {
                            "url": f"{base_url}/audio.wav",
                            "mimeType": "audio/wav",
                            "sampleRate": 24000,
                            "channels": 1,
                            "bitsPerSample": 16,
                        },
                    }
                ).encode()
            )

        def do_GET(self):
            requests.append((self.command, self.path, dict(self.headers), None))
            if self.path == "/health":
                self.reply(b"ok")
                return
            try:
                if scenario == "slow_headers_body":
                    time.sleep(0.2)
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.flush()
                    time.sleep(0.2)
                    self.wfile.write(body)
                elif scenario == "slow_header_progress":
                    self.wfile.write(b"HTTP/1.1 200 OK\r\nX-Progress: ")
                    for _ in range(10):
                        self.wfile.write(b"x")
                        self.wfile.flush()
                        time.sleep(0.05)
                    self.wfile.write(f"\r\nContent-Length: {len(body)}\r\n\r\n".encode() + body)
                else:
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    for offset in range(0, len(body), 8):
                        time.sleep(0.05)
                        self.wfile.write(body[offset : offset + 8])
                        self.wfile.flush()
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ssl.SSLEOFError):
                self.close_connection = True

    server = HTTPServer(("127.0.0.1", 0), Handler)
    server.socket = server_context.wrap_socket(server.socket, server_side=True)
    base_url = f"https://127.0.0.1:{server.server_port}"
    worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02})
    worker.start()
    try:
        yield base_url, client_context, requests
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)
        assert not worker.is_alive()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "worker", "async"])
@pytest.mark.parametrize("scenario", ["slow_headers_body", "slow_body_progress", "slow_header_progress"])
async def test_cooperative_deadline_rejects_late_audio_without_retry_or_client_closure(slow_speech_peer, mode):
    base_url, client_context, requests = slow_speech_peer
    responses = []
    completed = []

    def log_event(details):
        if details.get("log_event_type") == "post_api_call":
            completed.append(details)

    async def record_response(response):
        responses.append(response)

    kwargs = {
        "model": "flowspeech/flowspeech-tts",
        "input": "Hello",
        "api_key": "synthetic-loopback-only",
        "api_base": base_url,
        "request_id": REQUEST_ID,
        "timeout": BUDGET,
        "max_retries": 2,
        "logger_fn": log_event,
    }
    options = {
        "verify": client_context,
        "trust_env": False,
        "limits": httpx.Limits(max_connections=1, max_keepalive_connections=1),
    }
    if mode == "async":
        async with httpx.AsyncClient(**options, event_hooks={"response": [record_response]}) as client:
            with pytest.raises(litellm.Timeout, match="total deadline") as error:
                await asyncio.wait_for(litellm.aspeech(**kwargs, client=client), timeout=5)
            assert not client.is_closed
            assert responses and all(response.is_closed for response in responses)
            assert (await client.get(f"{base_url}/health", timeout=3)).content == b"ok"
    else:
        with httpx.Client(**options, event_hooks={"response": [responses.append]}) as client:
            if mode == "worker":
                with ThreadPoolExecutor(max_workers=1) as pool:
                    result = pool.submit(litellm.speech, **kwargs, client=client)
                    with pytest.raises(FlowSpeechException, match="total deadline") as error:
                        result.result(timeout=5)
            else:
                with pytest.raises(FlowSpeechException, match="total deadline") as error:
                    litellm.speech(**kwargs, client=client)
            assert not client.is_closed
            assert responses and all(response.is_closed for response in responses)
            assert client.get(f"{base_url}/health", timeout=3).content == b"ok"

    assert REQUEST_ID in str(error.value)
    assert JOB_ID in str(error.value)
    assert [(method, path) for method, path, _, _ in requests] == [
        ("POST", "/api/v1/tts/jobs"),
        ("GET", "/audio.wav"),
        ("GET", "/health"),
    ]
    assert httpx.Headers(requests[0][2])["Idempotency-Key"] == REQUEST_ID
    assert requests[0][3] == {"input": {"mode": "single", "text": "Hello", "voiceName": "Kore"}}
    assert "authorization" not in httpx.Headers(requests[1][2])
    assert completed == []
