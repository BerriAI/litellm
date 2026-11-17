"""The destination named in audit_log_callbacks receives a team or key update with the state before as an object."""

import json
import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from typing import Final

import pytest
from _s3_v2_support import BUCKET, PREFIX, RecordingS3Sink, s3_config
from integration._support.client import Gateway, JsonValue, Scenario, eventually, object_value
from integration._support.ports import allocate_port
from integration._support.process import OwnedProxy, owned_proxy_process
from integration._support.wire import wire_server

pytestmark: Final = pytest.mark.shares_worker_state

PER_KIND: Final = 15
AUDIT_PREFIX: Final = f"/{BUCKET}/{PREFIX}/audit_logs/"
STORED_SQL: Final = """
SELECT id, jsonb_typeof(before_value) AS before_type, before_value FROM "LiteLLM_AuditLog"
WHERE object_id = %s AND action = 'updated'
"""


@contextmanager
def _proxy_auditing_to(gateway: Gateway, directory: Path, sink_url: str) -> Iterator[OwnedProxy]:
    assert os.environ.get("LITELLM_LICENSE"), "LITELLM_LICENSE must be set: audit logs are an enterprise feature"
    credentials: Final[dict[str, JsonValue]] = {
        "s3_bucket_name": BUCKET,
        "s3_region_name": "us-east-1",
        "s3_path": PREFIX,
        "s3_aws_access_key_id": "AKIAIOSFODNN7EXAMPLE",
        "s3_aws_secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        "s3_endpoint_url": sink_url,
    }
    settings: Final[dict[str, JsonValue]] = {
        "store_audit_logs": True,
        "audit_log_callbacks": ["s3_v2"],
        "s3_audit_callback_params": credentials,
    }
    with owned_proxy_process(
        gateway,
        directory,
        {"DEFAULT_S3_FLUSH_INTERVAL_SECONDS": "1"},
        config=s3_config(directory, sink_url, {}, settings),
        workers=2,
    ) as owned:
        yield owned


def _updates(scenario: Scenario, per_kind: int) -> tuple[tuple[str, dict[str, JsonValue], str], ...]:
    teams: Final = tuple(scenario.team() for _ in range(per_kind))
    tokens: Final = tuple(sha256(scenario.key().encode()).hexdigest() for _ in range(per_kind))
    return (
        *(("/team/update", {"team_id": team, "max_budget": 7}, team) for team in teams),
        *(("/key/update", {"key": token, "max_budget": 3}, token) for token in tokens),
    )


def _send_at_once(candidate: Gateway, updates: Sequence[tuple[str, dict[str, JsonValue], str]]) -> list[int]:
    with ThreadPoolExecutor(len(updates)) as pool:
        return list(pool.map(lambda update: candidate.request("POST", update[0], update[1]).status_code, updates))


def _delivered_updates(sink: RecordingS3Sink) -> list[dict[str, JsonValue]]:
    audits: Final = (
        object_value(json.loads(body)) for target, body in sink.objects().items() if target.startswith(AUDIT_PREFIX)
    )
    return [audit for audit in audits if audit["action"] == "updated"]


def _delivered_for(sink: RecordingS3Sink, object_ids: Sequence[str]) -> Mapping[str, list[dict[str, JsonValue]]]:
    delivered: Final = _delivered_updates(sink)
    return {identity: [audit for audit in delivered if audit["object_id"] == identity] for identity in object_ids}


def _stored(candidate: Gateway, object_id: str) -> dict[str, JsonValue]:
    return eventually(lambda: candidate.rows(STORED_SQL, (object_id,)), lambda rows: len(rows) == 1, seconds=30)[0]


def test_the_audit_destination_receives_a_team_and_a_key_update_with_the_state_before_as_stored(
    gateway: Gateway, tmp_path: Path
) -> None:
    sink: Final = RecordingS3Sink(delay_seconds=0.05)
    with (
        wire_server(sink.respond) as bucket,
        _proxy_auditing_to(gateway, tmp_path, bucket.url) as owned,
        owned.gateway.scenario() as scenario,
    ):
        updates: Final = _updates(scenario, 1)
        object_ids: Final = [object_id for _, _, object_id in updates]
        assert _send_at_once(owned.gateway, updates) == [200, 200]
        stored: Final = [_stored(owned.gateway, object_id) for object_id in object_ids]

        delivered: Final = eventually(
            lambda: _delivered_for(sink, object_ids),
            lambda found: all(len(audits) == 1 for audits in found.values()),
            seconds=30,
        )

    received: Final = [delivered[object_id][0] for object_id in object_ids]
    assert [audit["id"] for audit in received] == [row["id"] for row in stored]
    assert [json.loads(str(audit["before_value"])) for audit in received] == [row["before_value"] for row in stored]
    assert [row["before_type"] for row in stored] == ["object", "object"], stored


@pytest.mark.timeout(300)
def test_updates_sent_while_the_audit_destination_is_down_all_answer_and_no_row_is_delivered_twice(
    gateway: Gateway, tmp_path: Path, record_property: Callable[[str, object], None]
) -> None:
    sink: Final = RecordingS3Sink(delay_seconds=0.05)
    port: Final = allocate_port()
    with (
        _proxy_auditing_to(gateway, tmp_path, f"http://127.0.0.1:{port}") as owned,
        owned.gateway.scenario() as scenario,
    ):
        updates: Final = _updates(scenario, PER_KIND)
        while_up, while_down = updates[::2], updates[1::2]
        after: Final = _updates(scenario, 1)
        every_id: Final = [object_id for _, _, object_id in (*updates, *after)]
        with wire_server(sink.respond, port=port):
            answered_up: Final = _send_at_once(owned.gateway, while_up)
            eventually(
                lambda: _delivered_for(sink, [object_id for _, _, object_id in while_up]),
                lambda found: all(len(audits) == 1 for audits in found.values()),
                seconds=30,
            )
        answered_down: Final = _send_at_once(owned.gateway, while_down)
        stored_down: Final = [_stored(owned.gateway, object_id) for _, _, object_id in while_down]
        with wire_server(sink.respond, port=port):
            answered_after: Final = _send_at_once(owned.gateway, after)
            eventually(
                lambda: _delivered_for(sink, [object_id for _, _, object_id in after]),
                lambda found: all(len(audits) == 1 for audits in found.values()),
                seconds=30,
            )
            delivered: Final = _delivered_for(sink, every_id)
        stored: Final = [_stored(owned.gateway, object_id) for object_id in every_id]

    record_property("delivered_of_those_sent_while_down", sum(len(delivered[o]) for _, _, o in while_down))
    assert (answered_up, answered_down, answered_after) == ([200] * PER_KIND, [200] * PER_KIND, [200, 200])
    assert [row["before_type"] for row in (*stored_down, *stored)] == ["object"] * (PER_KIND + len(every_id))
    assert [object_id for object_id, audits in delivered.items() if len(audits) > 1] == []
    assert [
        json.loads(str(audits[0]["before_value"])) for object_id in every_id if (audits := delivered[object_id])
    ] == [row["before_value"] for object_id, row in zip(every_id, stored) if delivered[object_id]]
