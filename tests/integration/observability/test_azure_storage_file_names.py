import re
import uuid
from pathlib import Path
from typing import Final

from _azure_storage_support import (
    SINK_HOSTS,
    RecordingDataLakeSink,
    azure_storage_config,
    azure_storage_environment,
)
from _s3_v2_support import surface_reply
from integration._support.client import Gateway, eventually
from integration._support.process import owned_proxy
from integration._support.tls import server_context, write_self_signed_cert
from integration._support.wire import wire_server

ADLS_SAFE_FILE_NAME: Final = re.compile(r"^[A-Za-z0-9._+-]+\.json$")


def _responses_id(candidate: Gateway, model: str, key: str, marker: str) -> str:
    response: Final = candidate.request("POST", "/v1/responses", {"model": model, "input": marker}, key=key)
    assert response.status_code == 200, response.text
    return str(response.json()["id"])


def test_responses_ids_with_base64_padding_land_under_adls_safe_names(gateway: Gateway, tmp_path: Path) -> None:
    """A /v1/responses id is `resp_` plus base64 with `=` padding decided by the encoded length, so upstream ids
    of several lengths yield both `=` and `==` padded ids; each must land as a file the service accepts."""
    marker: Final = f"azure-{uuid.uuid4().hex[:8]}"
    sink: Final = RecordingDataLakeSink()
    cert, key = write_self_signed_cert(tmp_path, SINK_HOSTS)
    with (
        wire_server(surface_reply) as provider,
        wire_server(sink.respond, tls=server_context(cert, key), keep_alive=True) as store,
    ):
        environment: Final = {**azure_storage_environment(store.url, cert), "DEFAULT_FLUSH_INTERVAL_SECONDS": "1"}
        config: Final = azure_storage_config(tmp_path)
        with (
            owned_proxy(gateway, tmp_path, environment, config=config, workers=1) as candidate,
            candidate.scenario() as scenario,
        ):
            model: Final = scenario.model(api_base=provider.url + "/v1", api_key="synthetic-provider-key")
            api_key: Final = scenario.key(models=[model])
            answered: Final = tuple(
                _responses_id(candidate, model, api_key, f"{marker}-{'x' * extra}") for extra in range(6)
            )
            assert {response_id.count("=") for response_id in answered} >= {1, 2}, answered
            eventually(
                lambda: len(sink.stored()) + len(sink.unauthenticated_targets()),
                lambda settled: settled >= len(answered),
                seconds=60,
            )
            assert sink.unauthenticated_targets() == (), sink.unauthenticated_targets()
            assert frozenset(str(payload["id"]) for payload in sink.payloads().values()) == frozenset(answered), tuple(
                sink.stored()
            )
            names: Final = tuple(path.rsplit("/", 1)[1] for path in sink.stored())
            assert all(ADLS_SAFE_FILE_NAME.match(name) for name in names), names
            assert len(frozenset(names)) == len(answered), names
        assert provider.drain()
