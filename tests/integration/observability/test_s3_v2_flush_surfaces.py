import re
import uuid
from pathlib import Path
from typing import Final

import pytest
from _s3_v2_support import (
    BUCKET,
    PREFIX,
    RecordingS3Sink,
    collect_payloads,
    matched_ids,
    mixed_burst,
    s3_config,
    surface_reply,
)
from integration._support.client import Gateway
from integration._support.process import owned_proxy
from integration._support.wire import wire_server

PER_REQUEST_KEY: Final = re.compile(rf"^/{BUCKET}/{PREFIX}/\d{{4}}-\d{{2}}-\d{{2}}/.+\.json$")
BATCH_KEY: Final = re.compile(
    rf"^/{BUCKET}/{PREFIX}/\d{{4}}-\d{{2}}-\d{{2}}/batch_\d{{2}}-\d{{2}}-\d{{2}}_[0-9a-f]{{32}}\.jsonl$"
)


@pytest.mark.covers("other.observability.s3_v2.mixed_surface_burst_bounds_puts_one_object_per_response_id")
def test_s3_v2_mixed_surface_burst_bounds_puts_one_object_per_response_id(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "s3mix" + uuid.uuid4().hex[:8]
    sink: Final = RecordingS3Sink()
    with wire_server(surface_reply) as provider, wire_server(sink.respond) as bucket:
        config: Final = s3_config(tmp_path, bucket.url, {})
        with (
            owned_proxy(gateway, tmp_path, {"DEFAULT_S3_FLUSH_INTERVAL_SECONDS": "3"}, config=config) as candidate,
            candidate.scenario() as scenario,
        ):
            openai_model: Final = scenario.model(api_base=provider.url + "/v1", api_key="synthetic-provider-key")
            anthropic_model: Final = scenario.model(
                model="anthropic/claude-sonnet-4-5-20250929", api_base=provider.url, api_key="synthetic-provider-key"
            )
            key: Final = scenario.key(models=[openai_model, anthropic_model])
            answered: Final = mixed_burst(candidate, openai_model, anthropic_model, key, marker)
            payloads: Final = collect_payloads(sink, len(answered))
            targets: Final = tuple(sink.objects())
    assert sum(1 for r in provider.drain() if r.method == "POST") == 48
    assert sink.peak <= 16, f"peak concurrent PUTs {sink.peak} exceeded the default bound"
    assert all(PER_REQUEST_KEY.match(target) for target in targets), list(targets)
    assert len(targets) == 48
    assert matched_ids(payloads, answered)


@pytest.mark.covers("other.observability.s3_v2.mixed_surface_batch_writes_ndjson_lines_per_response_id")
def test_s3_v2_mixed_surface_batch_writes_ndjson_lines_per_response_id(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "s3mixb" + uuid.uuid4().hex[:8]
    sink: Final = RecordingS3Sink()
    with wire_server(surface_reply) as provider, wire_server(sink.respond) as bucket:
        config: Final = s3_config(tmp_path, bucket.url, {"s3_batch_file_upload": True})
        with (
            owned_proxy(gateway, tmp_path, {"DEFAULT_S3_FLUSH_INTERVAL_SECONDS": "3"}, config=config) as candidate,
            candidate.scenario() as scenario,
        ):
            openai_model: Final = scenario.model(api_base=provider.url + "/v1", api_key="synthetic-provider-key")
            anthropic_model: Final = scenario.model(
                model="anthropic/claude-sonnet-4-5-20250929", api_base=provider.url, api_key="synthetic-provider-key"
            )
            key: Final = scenario.key(models=[openai_model, anthropic_model])
            answered: Final = mixed_burst(candidate, openai_model, anthropic_model, key, marker)
            payloads: Final = collect_payloads(sink, len(answered))
            targets: Final = tuple(sink.objects())
            puts: Final = bucket.drain()
    assert sum(1 for r in provider.drain() if r.method == "POST") == 48
    assert all(BATCH_KEY.match(target) for target in targets), list(targets)
    assert all(put.headers["content-type"] == "application/x-ndjson" for put in puts), [put.headers for put in puts]
    assert matched_ids(payloads, answered)
    assert len(payloads) == 48


@pytest.mark.covers("other.observability.s3_v2.sink_outage_mid_mixed_burst_recovers_every_response_id")
def test_s3_v2_sink_outage_mid_mixed_burst_recovers_every_response_id(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "s3mixo" + uuid.uuid4().hex[:8]
    sink: Final = RecordingS3Sink(fail_attempts=30, fail_status=503, delay_seconds=0.2)
    with wire_server(surface_reply) as provider, wire_server(sink.respond) as bucket:
        config: Final = s3_config(tmp_path, bucket.url, {})
        with (
            owned_proxy(gateway, tmp_path, {"DEFAULT_S3_FLUSH_INTERVAL_SECONDS": "3"}, config=config) as candidate,
            candidate.scenario() as scenario,
        ):
            openai_model: Final = scenario.model(api_base=provider.url + "/v1", api_key="synthetic-provider-key")
            anthropic_model: Final = scenario.model(
                model="anthropic/claude-sonnet-4-5-20250929", api_base=provider.url, api_key="synthetic-provider-key"
            )
            key: Final = scenario.key(models=[openai_model, anthropic_model])
            answered: Final = mixed_burst(candidate, openai_model, anthropic_model, key, marker)
            payloads: Final = collect_payloads(sink, len(answered), seconds=90)
    assert sum(1 for r in provider.drain() if r.method == "POST") == 48
    assert matched_ids(payloads, answered)
    assert len(payloads) == 48, "a stored id was overwritten or duplicated"
