"""Shared rig for the December 1 Google Vertex AI contracts, round 4 (`test_dec1_vertex4_*`).

Rounds 1 and 3 (`dec1_vertex_support.py`, `dec1_vertex3_support.py`) give a peer that is both the Vertex
model host (through `api_base`) and the OAuth token endpoint (through a generated service account's
`token_uri`). Round 4 adds what process level credential sources need:

- `owned_config`, which writes a copy of the shared proxy config with the response cache and retries off
  (and optional extra `litellm_settings`), for proxies that boot with their own environment.
- `MetadataPeer`, a loopback double of the GCE metadata server. google-auth reads its address from
  `GCE_METADATA_HOST` (metadata calls) and `GCE_METADATA_IP` (the `ping` that decides "am I on GCE"), so a
  proxy started with both pointing at the peer gets Application Default Credentials from it and never
  reaches `metadata.google.internal` or `169.254.169.254`. The same peer is also the Vertex model host.
- Small readers for bearer tokens, JWT headers and the base64 `X-Vertex-AI-Labels` header.

Nothing here builds a URL on a Google host.
"""

from __future__ import annotations

import base64
import json
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final
from urllib.parse import parse_qs, urlsplit

import yaml
from integration._support.client import JSON_OBJECT, object_value
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.providers.dec1_vertex_support import claude_echo, gemini_echo, path_of
from pydantic import JsonValue

METADATA_ROOT: Final = "/computeMetadata/v1/"
MDS_PROJECT: Final = "dec1-mds-project"
"""The project id the metadata double reports at `project/project-id`."""
MDS_EMAIL: Final = f"dec1-gce@{MDS_PROJECT}.iam.gserviceaccount.com"
ADC_PROJECT: Final = "dec1-adc-key-project"
"""The `project_id` inside the key file `GOOGLE_APPLICATION_CREDENTIALS` names."""
REF_PROJECT: Final = "dec1-envref-json-project"
"""The `project_id` inside the JSON text an `os.environ/` reference resolves to."""
VERTEX_LOCATION_ENV: Final = "asia-northeast1"
"""The value given to the `VERTEX_LOCATION` variable (not `VERTEXAI_LOCATION`) on the ADC proxy."""
LABELS_HEADER: Final = "x-vertex-ai-labels"
CLEARED_GOOGLE_ENVIRONMENT: Final = (
    "GOOGLE_APPLICATION_CREDENTIALS",
    "GOOGLE_CLOUD_PROJECT",
    "GCLOUD_PROJECT",
    "GCE_METADATA_HOST",
    "GCE_METADATA_ROOT",
    "GCE_METADATA_IP",
    "VERTEXAI_PROJECT",
    "VERTEXAI_LOCATION",
    "VERTEXAI_CREDENTIALS",
    "VERTEX_PROJECT",
    "VERTEX_LOCATION",
    "VERTEX_CREDENTIALS",
)
"""Variables an owned proxy drops from the inherited environment before applying its own overrides."""


def owned_config(directory: Path, **litellm_settings: JsonValue) -> Path:
    """A copy of `tests/integration/proxy_config.yaml` with `cache: false`, `num_retries: 0` and the given settings."""
    base: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    settings: Final = object_value(base["litellm_settings"])
    config: Final = directory / f"dec1-vertex4-{uuid.uuid4().hex}.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                **base,
                "litellm_settings": {**settings, "cache": False, **litellm_settings},
                "router_settings": {**object_value(base["router_settings"]), "num_retries": 0},
            }
        )
    )
    return config


def bearer(request: Request) -> str:
    return request.headers["authorization"].removeprefix("Bearer ")


def jwt_header(assertion: str) -> dict[str, JsonValue]:
    """The unverified header of a compact JWT (`alg`, `typ`, `kid`)."""
    head: Final = assertion.split(".")[0]
    return JSON_OBJECT.validate_json(base64.urlsafe_b64decode(head + "=" * (-len(head) % 4)))


def labels_header(request: Request) -> dict[str, JsonValue] | None:
    """The decoded `X-Vertex-AI-Labels` header of a recorded call, or None when the call has none."""
    encoded: Final = request.headers.get(LABELS_HEADER)
    if encoded is None:
        return None
    return JSON_OBJECT.validate_json(base64.b64decode(encoded))


def query_params(request: Request) -> dict[str, str]:
    return {name: values[0] for name, values in parse_qs(urlsplit(request.target).query).items()}


def vertex_echo(request: Request) -> Reply:
    """Answer Claude rawPredict calls in the Anthropic shape and everything else as Gemini."""
    if "/publishers/anthropic/" in path_of(request):
        return claude_echo(request)
    return gemini_echo(request)


# ---------------------------------------------------------------- the GCE metadata server double


@dataclass
class MetadataPeer:
    """A loopback metadata server that is also a Vertex model host.

    `GET /` answers 200 with `Metadata-Flavor: Google` (google-auth's `ping`), `project/project-id` answers
    `MDS_PROJECT` as text, `instance/service-accounts/default/` answers the account info, and
    `instance/service-accounts/default/token` issues `mds-token-1`, `mds-token-2`, ... valid for
    `expires_in` seconds. Any POST is a model call answered by `respond`.
    """

    wire: Wire
    expires_in: int = 3600
    issued: list[str] = field(default_factory=list)
    seen: list[Request] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def url(self) -> str:
        return self.wire.url.rstrip("/")

    @property
    def host(self) -> str:
        """`127.0.0.1:<port>`, the form `GCE_METADATA_HOST` and `GCE_METADATA_IP` take (no scheme)."""
        return urlsplit(self.wire.url).netloc

    def _drain(self) -> None:
        self.seen.extend(self.wire.drain())

    def metadata_calls(self) -> tuple[Request, ...]:
        self._drain()
        return tuple(request for request in self.seen if request.method == "GET")

    def token_calls(self) -> tuple[Request, ...]:
        return tuple(call for call in self.metadata_calls() if path_of(call).endswith("/token"))

    def model_calls(self) -> tuple[Request, ...]:
        self._drain()
        return tuple(request for request in self.seen if request.method == "POST")

    def answer_metadata(self, request: Request) -> Reply:
        flavor: Final = {"Metadata-Flavor": "Google"}
        path: Final = path_of(request)
        if path == "/":
            return Reply(body=b"", content_type="application/text", headers=flavor)
        if path == f"{METADATA_ROOT}project/project-id":
            return Reply(body=MDS_PROJECT.encode(), content_type="application/text", headers=flavor)
        if path == f"{METADATA_ROOT}instance/service-accounts/default/":
            info: Final = {
                "aliases": ["default"],
                "email": MDS_EMAIL,
                "scopes": ["https://www.googleapis.com/auth/cloud-platform"],
            }
            return Reply(body=json.dumps(info).encode(), headers=flavor)
        if path == f"{METADATA_ROOT}instance/service-accounts/default/token":
            with self._lock:
                token: Final = f"mds-token-{len(self.issued) + 1}"
                self.issued.append(token)
            body: Final = {"access_token": token, "expires_in": self.expires_in, "token_type": "Bearer"}
            return Reply(body=json.dumps(body).encode(), headers=flavor)
        return Reply(status=404, body=b"not found", content_type="application/text", headers=flavor)


@contextmanager
def metadata_peer(respond: Callable[[Request], Reply] = vertex_echo) -> Iterator[MetadataPeer]:
    holder: Final[list[MetadataPeer]] = []

    def answer(request: Request) -> Reply:
        if request.method == "GET":
            return holder[0].answer_metadata(request)
        return respond(request)

    with wire_server(answer) as wire:
        holder.append(MetadataPeer(wire))
        yield holder[0]


def metadata_environment(owned: MetadataPeer, gcloud_directory: Path) -> Mapping[str, str]:
    """Environment that sends google-auth's ADC search to the metadata double: no key file, an empty gcloud
    config directory, both metadata addresses on the peer, no mTLS to the metadata server, and the regional
    access boundary lookup (a call to an IAM host) off."""
    return {
        "CLOUDSDK_CONFIG": str(gcloud_directory),
        "GCE_METADATA_HOST": owned.host,
        "GCE_METADATA_IP": owned.host,
        "GCE_METADATA_MTLS_MODE": "none",
        "GOOGLE_AUTH_TRUST_BOUNDARY_ENABLED": "false",
    }
