import base64
import concurrent.futures
import hashlib
import http.client
import json
import resource
import sys
import time
import tracemalloc
from typing import Final, TypeAlias, TypedDict
from urllib.parse import urlsplit

from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.credentials import Credentials
from typing_extensions import ReadOnly

Json: TypeAlias = bool | int | float | str | list["Json"] | dict[str, "Json"] | None


class AudioInput(TypedDict):
    data: ReadOnly[str | bytes]
    format: ReadOnly[str]


class EncodedAudio(TypedDict):
    data: ReadOnly[str]
    format: ReadOnly[str]


def inputs(size: int, concurrency: int, encoding: str) -> tuple[AudioInput, ...]:
    return tuple(
        {"data": b"a" * size if encoding == "raw" else "A" * size, "format": "wav"} for _ in range(concurrency)
    )


def encode(audio: AudioInput) -> EncodedAudio:
    data: Final = audio["data"]
    return {
        "data": base64.b64encode(data).decode("ascii") if isinstance(data, bytes) else data,
        "format": audio["format"],
    }


def transform(audio: EncodedAudio) -> dict[str, Json]:
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"audio": {"format": audio["format"], "source": {"bytes": audio["data"]}}},
                    {"text": "Transcribe the audio. Respond with only the transcript."},
                ],
            }
        ],
        "system": [{"text": "You are a transcription assistant."}],
        "inferenceConfig": {"maxTokens": 4096},
    }


def stats() -> tuple[float, int]:
    usage: Final = resource.getrusage(resource.RUSAGE_SELF)
    return time.process_time(), usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024)


def start(measurement: str) -> tuple[float, int]:
    if measurement == "allocation":
        tracemalloc.start()
    return stats()


def finish(measurement: str) -> tuple[float, int, int]:
    return (*stats(), tracemalloc.get_traced_memory()[1] if measurement == "allocation" else 0)


def send(url: str, body: bytes, headers: dict[str, str], digest: str) -> None:
    target: Final = urlsplit(url)
    assert target.hostname is not None
    connection: Final = http.client.HTTPConnection(target.hostname, target.port, timeout=120)
    connection.request("POST", target.path, body=body, headers=headers)
    response: Final = connection.getresponse()
    assert response.status == 200
    assert response.read().decode() == digest
    connection.close()


def run(audio: tuple[AudioInput, ...], url: str) -> list[float]:
    started: Final = time.perf_counter()
    encoded: Final = tuple(encode(item) for item in audio)
    extraction: Final = time.perf_counter()
    transformed: Final = tuple(transform(item) for item in encoded)
    transformation: Final = time.perf_counter()
    bodies: Final = tuple(
        json.dumps(item, separators=(",", ":"), ensure_ascii=False, sort_keys=True).encode() for item in transformed
    )
    preparation: Final = time.perf_counter()
    digests: Final = tuple(hashlib.sha256(body).hexdigest() for body in bodies)
    requests: Final = tuple(
        AWSRequest(method="POST", url=url, data=body, headers={"X-Amz-Content-SHA256": digest})
        for body, digest in zip(bodies, digests)
    )
    for request in requests:
        SigV4Auth(Credentials("benchmark", "benchmark"), "bedrock", "us-east-1").add_auth(request)
    signing: Final = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(bodies)) as executor:
        tuple(
            executor.map(
                lambda args: send(url, *args), zip(bodies, (dict(request.headers) for request in requests), digests)
            )
        )
    return [
        extraction - started,
        transformation - extraction,
        preparation - transformation,
        signing - preparation,
        time.perf_counter() - signing,
    ]
