import asyncio
import os
import time
import uuid
from collections.abc import AsyncIterator, Iterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import partial
from types import MappingProxyType
from typing import Final
from urllib.parse import unquote

import httpx
import openai
import pytest
from integration._support.client import Gateway, Scenario, eventually, gateway_from_environment, string_value
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.management._batch_file_caps import (
    COMPLETION_TEXT,
    DAY_SECONDS,
    DOWNLOADS,
    IN_GENERAL_SETTINGS,
    IN_KEY,
    IN_TEAM,
    MINUTE_SECONDS,
    MISSING_FILE,
    PROVIDER_KEY,
    PROVIDER_REJECTS,
    RECORDS,
    ROUTED_MODEL,
    THIS_KEY,
    UPLOADS,
    Signer,
    Timed,
    assert_download_limited,
    assert_forwarded_upload,
    assert_too_many_records,
    assert_upload_limited,
    batch_file,
    batch_line,
    caps_config,
    counters,
    download,
    downloads_seen,
    file_content,
    hashed,
    marker,
    provider,
    provider_environment,
    seen,
    signed_token,
    signer,
    timed,
    upload,
    uploads_seen,
    window_end,
)
from pydantic import JsonValue
from redis import Redis

pytestmark = pytest.mark.timeout(600)

YAML_RECORDS: Final = 3
YAML_UPLOADS: Final = 4
ROOMY: Final = 1000
DAY_ROOM_SECONDS: Final = 120
MINUTE_ROOM_SECONDS: Final = 30
INVALIDATION_CHANNEL: Final = "litellm_proxy.auth_cache_invalidation"
UPLOAD_ROUTES: Final = ("/v1/files", "/files", "/openai/v1/files")
DOWNLOAD_ROUTES: Final = ("/v1/files/{}/content", "/files/{}/content", "/openai/v1/files/{}/content")
ROTATIONS: Final = (
    pytest.param(0, id="openai-prefixed-route-last"),
    pytest.param(1, id="v1-route-last"),
    pytest.param(2, id="bare-route-last"),
)
LEVELS: Final = ("key", "team")
MANAGED: Final = MappingProxyType({"target_model_names": ROUTED_MODEL})
HOSTILE_IDS: Final = (
    pytest.param("f" * 5120, id="5kb-id"),
    pytest.param("file-{}%20x", id="percent-encoded-space"),
)
SLASHED_IDS: Final = (
    pytest.param("file-{}/x", id="slash"),
    pytest.param("file-{}%2Fx", id="percent-encoded-slash"),
)


@dataclass(frozen=True, slots=True)
class Rig:
    candidate: Gateway
    sibling: Gateway
    provider: Wire
    cache: Redis
    identity: Signer

    def gateways(self, count: int) -> tuple[Gateway, ...]:
        return tuple((self.candidate, self.sibling)[index % 2] for index in range(count))


def _subscribers(cache: Redis) -> int:
    return int(cache.pubsub_numsub(INVALIDATION_CHANNEL)[0][1])


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    directory: Final = tmp_path_factory.mktemp("batch_file_caps")
    identity: Final = signer()

    def jwks(_request: Request) -> Reply:
        return Reply(body=identity.jwks)

    with (
        gateway_from_environment() as gateway,
        wire_server(provider) as files,
        wire_server(jwks) as keys,
        Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"])) as cache,
    ):
        config: Final = caps_config(
            directory,
            files.url,
            {
                RECORDS: YAML_RECORDS,
                UPLOADS: YAML_UPLOADS,
                "enable_jwt_auth": True,
                "litellm_jwtauth": {"user_id_jwt_field": "sub", "user_id_upsert": True},
            },
        )
        environment: Final = {**provider_environment(files.url), "JWT_PUBLIC_KEY_URL": keys.url}
        subscribed: Final = _subscribers(cache)
        with (
            owned_proxy(gateway, directory, environment, config=config, workers=2) as candidate,
            owned_proxy(gateway, directory, environment, config=config) as sibling,
        ):
            eventually(partial(_subscribers, cache), lambda count: count >= subscribed + 3, seconds=60)
            yield Rig(candidate, sibling, files, cache, identity)


def _key(scenario: Scenario, team: str | None = None, **limits: JsonValue) -> str:
    return scenario.key(metadata=limits, **({"team_id": team} if team is not None else {}))


def _accepted(response: httpx.Response) -> str:
    assert response.status_code == 200, response.text
    return string_value(response.json()["id"])


def _statuses(responses: tuple[httpx.Response, ...]) -> list[int]:
    return [response.status_code for response in responses]


def _counts(rig: Rig, setting: str, holder: str) -> list[int]:
    return [counter.count for counter in counters(rig.cache, setting, holder)]


def _rotated(routes: tuple[str, str, str], first: int) -> tuple[str, ...]:
    return routes[first:] + routes[:first]


def test_yaml_record_limit_rejects_a_longer_batch_file_before_it_reaches_the_provider(rig: Rig) -> None:
    with rig.candidate.scenario() as scenario:
        key: Final = scenario.key()
        over: Final = marker()
        within: Final = marker()
        content: Final = batch_file(within, YAML_RECORDS)
        assert_too_many_records(
            upload(rig.candidate, key, batch_file(over, YAML_RECORDS + 1)), YAML_RECORDS, IN_GENERAL_SETTINGS
        )
        assert _accepted(upload(rig.sibling, key, content)) == f"file-{within}"
        requests: Final = rig.provider.drain()
        assert seen(requests, over) == ()
        (forwarded,) = uploads_seen(requests, within)
        assert_forwarded_upload(forwarded, content)


@pytest.mark.parametrize(
    ("key_limit", "team_limit", "limit", "source"),
    [
        pytest.param(2, None, 2, IN_KEY, id="key-below-yaml"),
        pytest.param(5, None, 5, IN_KEY, id="key-above-yaml"),
        pytest.param(None, 2, 2, IN_TEAM, id="team-below-yaml"),
        pytest.param(5, 4, 4, IN_TEAM, id="team-below-key"),
        pytest.param(2, 4, 2, IN_KEY, id="key-below-team"),
        pytest.param(None, 5, YAML_RECORDS, IN_GENERAL_SETTINGS, id="yaml-below-team"),
    ],
)
def test_record_limit_is_the_lower_of_the_key_level_and_team_limits(
    rig: Rig, key_limit: int | None, team_limit: int | None, limit: int, source: str
) -> None:
    with rig.candidate.scenario() as scenario:
        team: Final = scenario.team(metadata={RECORDS: team_limit}) if team_limit is not None else None
        key: Final = _key(scenario, team, **({RECORDS: key_limit} if key_limit is not None else {}))
        over: Final = marker()
        within: Final = marker()
        content: Final = batch_file(within, limit)
        assert_too_many_records(upload(rig.candidate, key, batch_file(over, limit + 1)), limit, source)
        assert _accepted(upload(rig.sibling, key, content)) == f"file-{within}"
        requests: Final = rig.provider.drain()
        assert seen(requests, over) == ()
        (forwarded,) = uploads_seen(requests, within)
        assert_forwarded_upload(forwarded, content)


@pytest.mark.parametrize(
    ("separator", "ending", "records"),
    [
        pytest.param(b"\n\n\n", b"\n\n", YAML_RECORDS, id="blank-lines-at-the-limit"),
        pytest.param(b"\n \n", b"\n\t\n", YAML_RECORDS + 1, id="blank-lines-over-the-limit"),
        pytest.param(b"\r\n", b"", YAML_RECORDS, id="crlf-without-trailing-newline-at-the-limit"),
        pytest.param(b"\r\n", b"", YAML_RECORDS + 1, id="crlf-without-trailing-newline-over-the-limit"),
    ],
)
def test_record_limit_counts_request_lines_not_blank_lines_or_line_endings(
    rig: Rig, separator: bytes, ending: bytes, records: int
) -> None:
    with rig.candidate.scenario() as scenario:
        key: Final = scenario.key()
        mark: Final = marker()
        content: Final = batch_file(mark, records, separator, ending)
        response: Final = upload(rig.candidate, key, content)
        requests: Final = uploads_seen(rig.provider.drain(), mark)
        if records > YAML_RECORDS:
            assert_too_many_records(response, YAML_RECORDS, IN_GENERAL_SETTINGS)
            assert requests == ()
            return
        assert _accepted(response) == f"file-{mark}"
        (forwarded,) = requests
        assert_forwarded_upload(forwarded, content)


def test_only_batch_purpose_uploads_are_limited_and_counted(rig: Rig) -> None:
    with rig.candidate.scenario() as scenario:
        key: Final = _key(scenario, **{RECORDS: 1, UPLOADS: 1})
        mark: Final = marker()
        day_ends: Final = window_end(DAY_SECONDS, DAY_ROOM_SECONDS)
        purposes: Final = ("user_data", "fine-tune", "user_data", "fine-tune")
        others: Final = tuple(
            upload(gateway, key, batch_file(mark, 5), purpose=purpose, filename="notes.jsonl")
            for gateway, purpose in zip(rig.gateways(4), purposes, strict=True)
        )
        assert _statuses(others) == [200] * 4, [response.text for response in others]
        assert _accepted(upload(rig.candidate, key, batch_file(mark, 1))) == f"file-{mark}"
        assert_upload_limited(
            timed(partial(upload, rig.sibling, key, batch_file(mark, 1))), day_ends, 1, THIS_KEY, IN_KEY
        )
        forwarded: Final = uploads_seen(rig.provider.drain(), mark)
        assert [request.body.count(b"\r\n\r\nbatch\r\n") for request in forwarded] == [0, 0, 0, 0, 1], forwarded
        assert _counts(rig, UPLOADS, hashed(key)) == [1]


def test_yaml_daily_upload_limit_counts_one_key_across_proxy_processes(rig: Rig) -> None:
    with rig.candidate.scenario() as scenario:
        key: Final = scenario.key()
        other_key: Final = scenario.key()
        mark: Final = marker()
        other_mark: Final = marker()
        content: Final = batch_file(mark, 1)
        day_ends: Final = window_end(DAY_SECONDS, DAY_ROOM_SECONDS)
        accepted: Final = tuple(upload(gateway, key, content) for gateway in rig.gateways(YAML_UPLOADS))
        assert _statuses(accepted) == [200] * YAML_UPLOADS, [response.text for response in accepted]
        for gateway in rig.gateways(2):
            assert_upload_limited(
                timed(partial(upload, gateway, key, content)), day_ends, YAML_UPLOADS, THIS_KEY, IN_GENERAL_SETTINGS
            )
        assert _accepted(upload(rig.candidate, other_key, batch_file(other_mark, 1))) == f"file-{other_mark}"
        requests: Final = rig.provider.drain()
        forwarded: Final = uploads_seen(requests, mark)
        assert len(forwarded) == YAML_UPLOADS, forwarded
        for request in forwarded:
            assert_forwarded_upload(request, content)
        assert len(uploads_seen(requests, other_mark)) == 1
        (counter,) = counters(rig.cache, UPLOADS, hashed(key))
        assert counter.count == YAML_UPLOADS, counter
        assert 0 < counter.ttl <= DAY_SECONDS, counter
        assert counter.name.endswith(
            f"litellm:file_usage:{UPLOADS}:key:{hashed(key)}::{int(day_ends) - DAY_SECONDS}"
        ), counter


@pytest.mark.parametrize("limit", [2, 6])
def test_key_daily_upload_limit_replaces_the_yaml_limit(rig: Rig, limit: int) -> None:
    with rig.candidate.scenario() as scenario:
        key: Final = _key(scenario, **{UPLOADS: limit})
        mark: Final = marker()
        content: Final = batch_file(mark, 1)
        day_ends: Final = window_end(DAY_SECONDS, DAY_ROOM_SECONDS)
        accepted: Final = tuple(upload(gateway, key, content) for gateway in rig.gateways(limit))
        assert _statuses(accepted) == [200] * limit, [response.text for response in accepted]
        assert_upload_limited(timed(partial(upload, rig.candidate, key, content)), day_ends, limit, THIS_KEY, IN_KEY)
        assert len(uploads_seen(rig.provider.drain(), mark)) == limit
        assert _counts(rig, UPLOADS, hashed(key)) == [limit]


def test_upload_rejected_by_the_key_limit_does_not_use_a_team_slot(rig: Rig) -> None:
    with rig.candidate.scenario() as scenario:
        team: Final = scenario.team(metadata={UPLOADS: 5})
        tight_key: Final = _key(scenario, team, **{UPLOADS: 2})
        loose_key: Final = _key(scenario, team)
        tight_mark: Final = marker()
        loose_mark: Final = marker()
        day_ends: Final = window_end(DAY_SECONDS, DAY_ROOM_SECONDS)
        tight: Final = tuple(upload(gateway, tight_key, batch_file(tight_mark, 1)) for gateway in rig.gateways(2))
        assert _statuses(tight) == [200, 200], [response.text for response in tight]
        for gateway in rig.gateways(2):
            assert_upload_limited(
                timed(partial(upload, gateway, tight_key, batch_file(tight_mark, 1))), day_ends, 2, THIS_KEY, IN_KEY
            )
        loose: Final = tuple(upload(gateway, loose_key, batch_file(loose_mark, 1)) for gateway in rig.gateways(3))
        assert _statuses(loose) == [200, 200, 200], [response.text for response in loose]
        assert_upload_limited(
            timed(partial(upload, rig.sibling, loose_key, batch_file(loose_mark, 1))),
            day_ends,
            5,
            f"team {team}",
            IN_TEAM,
        )
        requests: Final = rig.provider.drain()
        assert len(uploads_seen(requests, tight_mark)) == 2
        assert len(uploads_seen(requests, loose_mark)) == 3
        assert _counts(rig, UPLOADS, hashed(tight_key)) == [2]
        assert _counts(rig, UPLOADS, hashed(loose_key)) == [3]
        assert _counts(rig, UPLOADS, f"team:{team}:") == [5]


def test_rejected_uploads_do_not_use_a_daily_upload_slot(rig: Rig) -> None:
    with rig.candidate.scenario() as scenario:
        key: Final = scenario.key()
        rejected_mark: Final = marker()
        mark: Final = marker()
        content: Final = batch_file(mark, 1)
        day_ends: Final = window_end(DAY_SECONDS, DAY_ROOM_SECONDS)
        assert_too_many_records(
            upload(rig.candidate, key, batch_file(rejected_mark, YAML_RECORDS + 1)), YAML_RECORDS, IN_GENERAL_SETTINGS
        )
        bad_expiry: Final = upload(
            rig.sibling,
            key,
            batch_file(rejected_mark, 1),
            fields={"expires_after[anchor]": "created_at", "expires_after[seconds]": "soon"},
        )
        assert bad_expiry.status_code == 400, bad_expiry.text
        assert "expires_after[seconds] must be a valid integer, got 'soon'" in bad_expiry.text
        invalid_line: Final = upload(rig.candidate, key, batch_line(rejected_mark, 0) + b"\n{not json\n")
        assert invalid_line.status_code == 400, invalid_line.text
        assert "Batch input file line 2 is not valid JSON" in invalid_line.text
        wrong_extension: Final = upload(rig.sibling, key, batch_file(rejected_mark, 1), filename="batch.txt")
        assert wrong_extension.status_code == 400, wrong_extension.text
        assert "Batch input files must be .jsonl files" in wrong_extension.text
        accepted: Final = tuple(upload(gateway, key, content) for gateway in rig.gateways(YAML_UPLOADS))
        assert _statuses(accepted) == [200] * YAML_UPLOADS, [response.text for response in accepted]
        assert_upload_limited(
            timed(partial(upload, rig.candidate, key, content)), day_ends, YAML_UPLOADS, THIS_KEY, IN_GENERAL_SETTINGS
        )
        requests: Final = rig.provider.drain()
        assert seen(requests, rejected_mark) == ()
        assert len(uploads_seen(requests, mark)) == YAML_UPLOADS
        assert _counts(rig, UPLOADS, hashed(key)) == [YAML_UPLOADS]


def test_callers_without_a_valid_key_are_rejected_before_anything_is_counted(rig: Rig) -> None:
    mark: Final = marker()
    unknown_key: Final = f"sk-{mark}"
    file_id: Final = f"file-{mark}"
    responses: Final = (
        rig.candidate.client.post(
            "/v1/files",
            data={"purpose": "batch"},
            files={"file": ("batch.jsonl", batch_file(mark, 1), "application/jsonl")},
        ),
        upload(rig.sibling, unknown_key, batch_file(mark, 1)),
        rig.candidate.client.get(f"/v1/files/{file_id}/content"),
        download(rig.sibling, unknown_key, file_id),
    )
    assert _statuses(responses) == [401, 401, 401, 401], [response.text for response in responses]
    assert seen(rig.provider.drain(), mark) == ()
    assert counters(rig.cache, UPLOADS, hashed(unknown_key)) == ()
    assert counters(rig.cache, DOWNLOADS, mark) == ()


def test_upload_the_provider_rejects_still_uses_a_daily_upload_slot(rig: Rig) -> None:
    with rig.candidate.scenario() as scenario:
        key: Final = _key(scenario, **{UPLOADS: 2})
        mark: Final = marker()
        content: Final = batch_file(mark, 1)
        day_ends: Final = window_end(DAY_SECONDS, DAY_ROOM_SECONDS)
        refused: Final = upload(rig.candidate, key, batch_file(f"{mark}-{PROVIDER_REJECTS}", 1))
        assert refused.status_code == 400, refused.text
        assert "The provider rejected this batch file." in refused.text
        assert _accepted(upload(rig.sibling, key, content)) == f"file-{mark}"
        assert_upload_limited(timed(partial(upload, rig.candidate, key, content)), day_ends, 2, THIS_KEY, IN_KEY)
        assert len(uploads_seen(rig.provider.drain(), mark)) == 2
        assert _counts(rig, UPLOADS, hashed(key)) == [2]


def test_openai_sdk_caller_sees_each_limit_as_a_status_error(rig: Rig) -> None:
    with rig.candidate.scenario() as scenario:
        key: Final = _key(scenario, **{UPLOADS: 2, DOWNLOADS: 2})
        mark: Final = marker()
        file_id: Final = f"file-{mark}"
        content: Final = batch_file(mark, 1)
        day_ends: Final = window_end(DAY_SECONDS, DAY_ROOM_SECONDS)
        minute_ends: Final = window_end(MINUTE_SECONDS, MINUTE_ROOM_SECONDS)
        with (
            httpx.Client(timeout=15, trust_env=False) as transport,
            openai.OpenAI(
                base_url=f"{str(rig.candidate.client.base_url).rstrip('/')}/v1",
                api_key=key,
                max_retries=0,
                http_client=transport,
            ) as client,
        ):
            with pytest.raises(openai.APIStatusError) as too_long:
                client.files.create(file=("batch.jsonl", batch_file(marker(), YAML_RECORDS + 1)), purpose="batch")
            assert_too_many_records(too_long.value.response, YAML_RECORDS, IN_GENERAL_SETTINGS)
            created: Final = tuple(
                client.files.create(file=("batch.jsonl", content), purpose="batch") for _ in range(2)
            )
            assert [file.id for file in created] == [file_id, file_id]
            upload_started: Final = time.time()
            with pytest.raises(openai.RateLimitError) as upload_limited:
                client.files.create(file=("batch.jsonl", content), purpose="batch")
            assert_upload_limited(
                Timed(upload_limited.value.response, upload_started, time.time()), day_ends, 2, THIS_KEY, IN_KEY
            )
            downloaded: Final = tuple(client.files.content(file_id).content for _ in range(2))
            assert downloaded == (file_content(file_id), file_content(file_id))
            download_started: Final = time.time()
            with pytest.raises(openai.RateLimitError) as download_limited:
                client.files.content(file_id)
            assert_download_limited(
                Timed(download_limited.value.response, download_started, time.time()),
                minute_ends,
                file_id,
                2,
                THIS_KEY,
                IN_KEY,
            )
        requests: Final = rig.provider.drain()
        assert len(uploads_seen(requests, mark)) == 2
        assert len(downloads_seen(requests, file_id)) == 2


async def test_async_openai_sdk_caller_sees_each_limit_as_a_status_error(rig: Rig) -> None:
    with rig.candidate.scenario() as scenario:
        key: Final = _key(scenario, **{UPLOADS: 2, DOWNLOADS: 2})
        mark: Final = marker()
        file_id: Final = f"file-{mark}"
        content: Final = batch_file(mark, 1)
        day_ends: Final = window_end(DAY_SECONDS, DAY_ROOM_SECONDS)
        minute_ends: Final = window_end(MINUTE_SECONDS, MINUTE_ROOM_SECONDS)
        async with (
            httpx.AsyncClient(timeout=15, trust_env=False) as transport,
            openai.AsyncOpenAI(
                base_url=f"{str(rig.sibling.client.base_url).rstrip('/')}/v1",
                api_key=key,
                max_retries=0,
                http_client=transport,
            ) as client,
        ):
            with pytest.raises(openai.APIStatusError) as too_long:
                await client.files.create(file=("batch.jsonl", batch_file(marker(), YAML_RECORDS + 1)), purpose="batch")
            assert_too_many_records(too_long.value.response, YAML_RECORDS, IN_GENERAL_SETTINGS)
            first: Final = await client.files.create(file=("batch.jsonl", content), purpose="batch")
            second: Final = await client.files.create(file=("batch.jsonl", content), purpose="batch")
            assert [first.id, second.id] == [file_id, file_id]
            upload_started: Final = time.time()
            with pytest.raises(openai.RateLimitError) as upload_limited:
                await client.files.create(file=("batch.jsonl", content), purpose="batch")
            assert_upload_limited(
                Timed(upload_limited.value.response, upload_started, time.time()), day_ends, 2, THIS_KEY, IN_KEY
            )
            first_download: Final = await client.files.content(file_id)
            second_download: Final = await client.files.content(file_id)
            assert [first_download.content, second_download.content] == [file_content(file_id), file_content(file_id)]
            download_started: Final = time.time()
            with pytest.raises(openai.RateLimitError) as download_limited:
                await client.files.content(file_id)
            assert_download_limited(
                Timed(download_limited.value.response, download_started, time.time()),
                minute_ends,
                file_id,
                2,
                THIS_KEY,
                IN_KEY,
            )
        requests: Final = rig.provider.drain()
        assert len(uploads_seen(requests, mark)) == 2
        assert len(downloads_seen(requests, file_id)) == 2


@pytest.mark.parametrize("first", ROTATIONS)
def test_every_upload_route_shares_one_daily_count(rig: Rig, first: int) -> None:
    with rig.candidate.scenario() as scenario:
        key: Final = _key(scenario, **{UPLOADS: 2})
        mark: Final = marker()
        content: Final = batch_file(mark, 1)
        allowed_first, allowed_second, limited = _rotated(UPLOAD_ROUTES, first)
        day_ends: Final = window_end(DAY_SECONDS, DAY_ROOM_SECONDS)
        assert _accepted(upload(rig.candidate, key, content, path=allowed_first)) == f"file-{mark}"
        assert _accepted(upload(rig.sibling, key, content, path=allowed_second)) == f"file-{mark}"
        assert_upload_limited(
            timed(partial(upload, rig.candidate, key, content, path=limited)), day_ends, 2, THIS_KEY, IN_KEY
        )
        assert len(uploads_seen(rig.provider.drain(), mark)) == 2
        assert _counts(rig, UPLOADS, hashed(key)) == [2]


def test_jwt_callers_are_counted_per_user_across_tokens(rig: Rig) -> None:
    subject: Final = f"caps-jwt-{uuid.uuid4().hex}"
    other_subject: Final = f"caps-jwt-{uuid.uuid4().hex}"
    with rig.candidate.scenario() as scenario:
        mark: Final = marker()
        other_mark: Final = marker()
        content: Final = batch_file(mark, 1)
        day_ends: Final = window_end(DAY_SECONDS, DAY_ROOM_SECONDS)
        first: Final = upload(rig.candidate, signed_token(rig.identity, subject), content)
        scenario.cleanups.callback(scenario.delete_user, subject)
        assert _accepted(first) == f"file-{mark}"
        rest: Final = tuple(
            upload(gateway, signed_token(rig.identity, subject), content) for gateway in rig.gateways(YAML_UPLOADS - 1)
        )
        assert _statuses(rest) == [200] * (YAML_UPLOADS - 1), [response.text for response in rest]
        assert_upload_limited(
            timed(partial(upload, rig.sibling, signed_token(rig.identity, subject), content)),
            day_ends,
            YAML_UPLOADS,
            f"user {subject}",
            IN_GENERAL_SETTINGS,
        )
        other: Final = upload(rig.candidate, signed_token(rig.identity, other_subject), batch_file(other_mark, 1))
        scenario.cleanups.callback(scenario.delete_user, other_subject)
        assert _accepted(other) == f"file-{other_mark}"
        requests: Final = rig.provider.drain()
        assert len(uploads_seen(requests, mark)) == YAML_UPLOADS
        assert len(uploads_seen(requests, other_mark)) == 1
        assert _counts(rig, UPLOADS, f"user:{subject}:") == [YAML_UPLOADS]


def test_key_download_limit_counts_one_file_per_key_across_proxy_processes(rig: Rig) -> None:
    with rig.candidate.scenario() as scenario:
        key: Final = _key(scenario, **{DOWNLOADS: 3})
        other_key: Final = _key(scenario, **{DOWNLOADS: 3})
        file_id: Final = f"file-{marker()}"
        other_file: Final = f"file-{marker()}"
        minute_ends: Final = window_end(MINUTE_SECONDS, MINUTE_ROOM_SECONDS)
        allowed: Final = tuple(download(gateway, key, file_id) for gateway in rig.gateways(3))
        assert _statuses(allowed) == [200, 200, 200], [response.text for response in allowed]
        assert [response.content for response in allowed] == [file_content(file_id)] * 3
        for gateway in rig.gateways(2):
            assert_download_limited(
                timed(partial(download, gateway, key, file_id)), minute_ends, file_id, 3, THIS_KEY, IN_KEY
            )
        same_key_other_file: Final = download(rig.candidate, key, other_file)
        assert (same_key_other_file.status_code, same_key_other_file.content) == (200, file_content(other_file))
        other_key_same_file: Final = download(rig.sibling, other_key, file_id)
        assert (other_key_same_file.status_code, other_key_same_file.content) == (200, file_content(file_id))
        requests: Final = rig.provider.drain()
        forwarded: Final = downloads_seen(requests, file_id)
        assert len(forwarded) == 4, forwarded
        assert {request.headers["authorization"] for request in forwarded} == {f"Bearer {PROVIDER_KEY}"}
        assert len(downloads_seen(requests, other_file)) == 1
        (counter,) = counters(rig.cache, DOWNLOADS, f"{hashed(key)}:{file_id}:")
        assert counter.count == 3, counter
        assert 0 < counter.ttl <= MINUTE_SECONDS, counter
        assert counter.name.endswith(
            f"litellm:file_usage:{DOWNLOADS}:key:{hashed(key)}:{file_id}:{int(minute_ends) - MINUTE_SECONDS}"
        ), counter


def test_team_download_limit_is_shared_by_every_key_on_the_team(rig: Rig) -> None:
    with rig.candidate.scenario() as scenario:
        team: Final = scenario.team(metadata={DOWNLOADS: 3})
        first_key: Final = _key(scenario, team)
        second_key: Final = _key(scenario, team)
        outside_key: Final = scenario.key()
        file_id: Final = f"file-{marker()}"
        minute_ends: Final = window_end(MINUTE_SECONDS, MINUTE_ROOM_SECONDS)
        allowed: Final = (
            download(rig.candidate, first_key, file_id),
            download(rig.sibling, first_key, file_id),
            download(rig.candidate, second_key, file_id),
        )
        assert _statuses(allowed) == [200, 200, 200], [response.text for response in allowed]
        for gateway, key in zip(rig.gateways(2), (second_key, first_key), strict=True):
            assert_download_limited(
                timed(partial(download, gateway, key, file_id)), minute_ends, file_id, 3, f"team {team}", IN_TEAM
            )
        outside: Final = download(rig.sibling, outside_key, file_id)
        assert (outside.status_code, outside.content) == (200, file_content(file_id)), outside.text
        assert len(downloads_seen(rig.provider.drain(), file_id)) == 4
        assert _counts(rig, DOWNLOADS, f"team:{team}:{file_id}:") == [3]


def test_downloads_are_unlimited_when_no_limit_is_set(rig: Rig) -> None:
    with rig.candidate.scenario() as scenario:
        key: Final = scenario.key()
        file_id: Final = f"file-{marker()}"
        responses: Final = tuple(download(gateway, key, file_id) for gateway in rig.gateways(12))
        assert _statuses(responses) == [200] * 12, [response.text for response in responses]
        assert {response.content for response in responses} == {file_content(file_id)}
        assert len(downloads_seen(rig.provider.drain(), file_id)) == 12
        assert counters(rig.cache, DOWNLOADS, file_id) == ()


def test_download_the_provider_cannot_find_still_uses_a_slot(rig: Rig) -> None:
    with rig.candidate.scenario() as scenario:
        key: Final = _key(scenario, **{DOWNLOADS: 2})
        file_id: Final = f"{MISSING_FILE}{marker()}"
        minute_ends: Final = window_end(MINUTE_SECONDS, MINUTE_ROOM_SECONDS)
        missing: Final = tuple(download(gateway, key, file_id) for gateway in rig.gateways(2))
        assert _statuses(missing) == [404, 404], [response.text for response in missing]
        assert f"No such File object: {file_id}" in missing[0].text
        assert_download_limited(
            timed(partial(download, rig.candidate, key, file_id)), minute_ends, file_id, 2, THIS_KEY, IN_KEY
        )
        assert len(downloads_seen(rig.provider.drain(), file_id)) == 2


@pytest.mark.parametrize("first", ROTATIONS)
def test_every_download_route_shares_one_count_per_file(rig: Rig, first: int) -> None:
    with rig.candidate.scenario() as scenario:
        key: Final = _key(scenario, **{DOWNLOADS: 2})
        file_id: Final = f"file-{marker()}"
        allowed_first, allowed_second, limited = _rotated(DOWNLOAD_ROUTES, first)
        minute_ends: Final = window_end(MINUTE_SECONDS, MINUTE_ROOM_SECONDS)
        allowed: Final = (
            download(rig.candidate, key, file_id, route=allowed_first),
            download(rig.sibling, key, file_id, route=allowed_second),
        )
        assert _statuses(allowed) == [200, 200], [response.text for response in allowed]
        assert_download_limited(
            timed(partial(download, rig.candidate, key, file_id, route=limited)),
            minute_ends,
            file_id,
            2,
            THIS_KEY,
            IN_KEY,
        )
        assert len(downloads_seen(rig.provider.drain(), file_id)) == 2


def test_model_routed_files_are_counted_under_the_id_the_caller_uses(rig: Rig) -> None:
    with rig.candidate.scenario() as scenario:
        key: Final = _key(scenario, **{DOWNLOADS: 2})
        mark: Final = marker()
        provider_id: Final = f"file-{mark}"
        plain_id: Final = f"file-{marker()}"
        routed_id: Final = _accepted(upload(rig.candidate, key, batch_file(mark, 1), fields={"model": ROUTED_MODEL}))
        assert routed_id != provider_id
        minute_ends: Final = window_end(MINUTE_SECONDS, MINUTE_ROOM_SECONDS)
        routed: Final = tuple(download(gateway, key, routed_id) for gateway in rig.gateways(2))
        assert _statuses(routed) == [200, 200], [response.text for response in routed]
        assert {response.content for response in routed} == {file_content(provider_id)}
        assert_download_limited(
            timed(partial(download, rig.candidate, key, routed_id)), minute_ends, routed_id, 2, THIS_KEY, IN_KEY
        )
        by_query: Final = tuple(
            download(gateway, key, plain_id, params={"model": ROUTED_MODEL}) for gateway in rig.gateways(2)
        )
        assert _statuses(by_query) == [200, 200], [response.text for response in by_query]
        assert_download_limited(
            timed(partial(download, rig.sibling, key, plain_id, params={"model": ROUTED_MODEL})),
            minute_ends,
            plain_id,
            2,
            THIS_KEY,
            IN_KEY,
        )
        requests: Final = rig.provider.drain()
        assert len(uploads_seen(requests, mark)) == 1
        assert len(downloads_seen(requests, provider_id)) == 2
        assert len(downloads_seen(requests, plain_id)) == 2


def test_managed_files_are_capped_and_counted_under_their_unified_id(rig: Rig) -> None:
    with rig.candidate.scenario() as scenario:
        key: Final = _key(scenario, **{UPLOADS: 1, DOWNLOADS: 2})
        mark: Final = marker()
        provider_id: Final = f"file-{mark}"
        rejected_mark: Final = marker()
        limited_mark: Final = marker()
        assert_too_many_records(
            upload(rig.candidate, key, batch_file(rejected_mark, YAML_RECORDS + 1), fields=MANAGED),
            YAML_RECORDS,
            IN_GENERAL_SETTINGS,
        )
        day_ends: Final = window_end(DAY_SECONDS, DAY_ROOM_SECONDS)
        unified_id: Final = _accepted(upload(rig.sibling, key, batch_file(mark, 1), fields=MANAGED))
        assert unified_id != provider_id
        assert_upload_limited(
            timed(partial(upload, rig.candidate, key, batch_file(limited_mark, 1), fields=MANAGED)),
            day_ends,
            1,
            THIS_KEY,
            IN_KEY,
        )
        minute_ends: Final = window_end(MINUTE_SECONDS, MINUTE_ROOM_SECONDS)
        served: Final = tuple(download(gateway, key, unified_id) for gateway in rig.gateways(2))
        assert _statuses(served) == [200, 200], [response.text for response in served]
        assert {response.content for response in served} == {file_content(provider_id)}
        assert_download_limited(
            timed(partial(download, rig.sibling, key, unified_id)), minute_ends, unified_id, 2, THIS_KEY, IN_KEY
        )
        requests: Final = rig.provider.drain()
        assert len(uploads_seen(requests, mark)) == 1
        assert seen(requests, rejected_mark) == ()
        assert seen(requests, limited_mark) == ()
        assert len(downloads_seen(requests, provider_id)) == 2
        assert _counts(rig, UPLOADS, hashed(key)) == [1]
        assert _counts(rig, DOWNLOADS, f"{hashed(key)}:{unified_id}:") == [2]


@pytest.mark.parametrize("shape", HOSTILE_IDS)
def test_hostile_file_ids_are_counted_as_the_provider_receives_them(rig: Rig, shape: str) -> None:
    with rig.candidate.scenario() as scenario:
        key: Final = _key(scenario, **{DOWNLOADS: 2})
        sent: Final = shape.format(marker())
        counted: Final = unquote(sent)
        minute_ends: Final = window_end(MINUTE_SECONDS, MINUTE_ROOM_SECONDS)
        served: Final = tuple(download(gateway, key, sent) for gateway in rig.gateways(2))
        assert _statuses(served) == [200, 200], [response.text for response in served]
        assert {response.content for response in served} == {file_content(counted)}
        assert_download_limited(
            timed(partial(download, rig.candidate, key, sent)), minute_ends, counted, 2, THIS_KEY, IN_KEY
        )
        assert len(downloads_seen(rig.provider.drain(), sent)) == 2
        assert _counts(rig, DOWNLOADS, f"{hashed(key)}:{counted}:") == [2]


@pytest.mark.parametrize("shape", SLASHED_IDS)
def test_a_file_id_with_a_slash_never_reaches_the_file_route_or_a_counter(rig: Rig, shape: str) -> None:
    with rig.candidate.scenario() as scenario:
        key: Final = _key(scenario, **{DOWNLOADS: 2})
        mark: Final = marker()
        answers: Final = tuple(download(gateway, key, shape.format(mark)) for gateway in rig.gateways(3))
        assert _statuses(answers) == [401, 401, 401], [response.text for response in answers]
        assert seen(rig.provider.drain(), mark) == ()
        assert counters(rig.cache, DOWNLOADS, hashed(key)) == ()


def _every_limit(value: JsonValue) -> dict[str, JsonValue]:
    return {RECORDS: value, UPLOADS: value, DOWNLOADS: value}


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize(
    "metadata",
    [
        pytest.param(_every_limit(0), id="zero"),
        pytest.param(_every_limit(-1), id="negative"),
        pytest.param(_every_limit(""), id="empty-string"),
        pytest.param(_every_limit("abc"), id="word"),
        pytest.param(_every_limit([5]), id="list"),
        pytest.param(_every_limit({"a": 1}), id="object"),
        pytest.param(_every_limit(5.5), id="fraction"),
        pytest.param(_every_limit("x" * 5120), id="5kb-string"),
        pytest.param(_every_limit(None), id="null"),
        pytest.param({}, id="empty-metadata"),
    ],
)
def test_malformed_limits_are_ignored_and_the_yaml_limits_still_apply(
    rig: Rig, metadata: Mapping[str, JsonValue], level: str
) -> None:
    with rig.candidate.scenario() as scenario:
        team: Final = scenario.team(metadata=dict(metadata)) if level == "team" else None
        key: Final = _key(scenario, team, **(metadata if level == "key" else {}))
        over: Final = marker()
        mark: Final = marker()
        file_id: Final = f"file-{mark}"
        content: Final = batch_file(mark, YAML_RECORDS)
        day_ends: Final = window_end(DAY_SECONDS, DAY_ROOM_SECONDS)
        assert_too_many_records(
            upload(rig.candidate, key, batch_file(over, YAML_RECORDS + 1)), YAML_RECORDS, IN_GENERAL_SETTINGS
        )
        accepted: Final = tuple(upload(gateway, key, content) for gateway in rig.gateways(YAML_UPLOADS))
        assert _statuses(accepted) == [200] * YAML_UPLOADS, [response.text for response in accepted]
        assert_upload_limited(
            timed(partial(upload, rig.candidate, key, content)), day_ends, YAML_UPLOADS, THIS_KEY, IN_GENERAL_SETTINGS
        )
        downloads: Final = tuple(download(gateway, key, file_id) for gateway in rig.gateways(6))
        assert _statuses(downloads) == [200] * 6, [response.text for response in downloads]
        requests: Final = rig.provider.drain()
        assert seen(requests, over) == ()
        assert len(uploads_seen(requests, mark)) == YAML_UPLOADS
        assert len(downloads_seen(requests, file_id)) == 6


@pytest.mark.parametrize("level", LEVELS)
def test_numeric_string_limits_are_honored(rig: Rig, level: str) -> None:
    with rig.candidate.scenario() as scenario:
        team: Final = scenario.team(metadata=_every_limit("2")) if level == "team" else None
        key: Final = _key(scenario, team, **(_every_limit("2") if level == "key" else {}))
        holder: Final = THIS_KEY if level == "key" else f"team {team}"
        source: Final = IN_KEY if level == "key" else IN_TEAM
        over: Final = marker()
        mark: Final = marker()
        file_id: Final = f"file-{mark}"
        content: Final = batch_file(mark, 2)
        day_ends: Final = window_end(DAY_SECONDS, DAY_ROOM_SECONDS)
        minute_ends: Final = window_end(MINUTE_SECONDS, MINUTE_ROOM_SECONDS)
        assert_too_many_records(upload(rig.candidate, key, batch_file(over, 3)), 2, source)
        accepted: Final = tuple(upload(gateway, key, content) for gateway in rig.gateways(2))
        assert _statuses(accepted) == [200, 200], [response.text for response in accepted]
        assert_upload_limited(timed(partial(upload, rig.candidate, key, content)), day_ends, 2, holder, source)
        downloads: Final = tuple(download(gateway, key, file_id) for gateway in rig.gateways(2))
        assert _statuses(downloads) == [200, 200], [response.text for response in downloads]
        assert_download_limited(
            timed(partial(download, rig.candidate, key, file_id)), minute_ends, file_id, 2, holder, source
        )
        requests: Final = rig.provider.drain()
        assert seen(requests, over) == ()
        assert len(uploads_seen(requests, mark)) == 2
        assert len(downloads_seen(requests, file_id)) == 2


def test_limited_key_still_reaches_chat_completions_and_file_metadata(rig: Rig) -> None:
    with rig.candidate.scenario() as scenario:
        key: Final = _key(scenario, **{UPLOADS: 1, DOWNLOADS: 1})
        mark: Final = marker()
        file_id: Final = f"file-{mark}"
        content: Final = batch_file(mark, 1)
        day_ends: Final = window_end(DAY_SECONDS, DAY_ROOM_SECONDS)
        minute_ends: Final = window_end(MINUTE_SECONDS, MINUTE_ROOM_SECONDS)
        assert _accepted(upload(rig.candidate, key, content)) == file_id
        assert_upload_limited(timed(partial(upload, rig.sibling, key, content)), day_ends, 1, THIS_KEY, IN_KEY)
        assert download(rig.candidate, key, file_id).status_code == 200
        assert_download_limited(
            timed(partial(download, rig.sibling, key, file_id)), minute_ends, file_id, 1, THIS_KEY, IN_KEY
        )
        for gateway in rig.gateways(4):
            completion: Final = gateway.chat(ROUTED_MODEL, key=key, text=mark)
            assert completion["choices"][0]["message"]["content"] == COMPLETION_TEXT, completion
            described: Final = gateway.request("GET", f"/v1/files/{file_id}", key=key)
            assert described.status_code == 200, described.text
            assert described.json()["id"] == file_id, described.text


def _lowered_probe(rig: Rig, key: str) -> tuple[httpx.Response, httpx.Response, httpx.Response]:
    file_id: Final = f"file-{marker()}"
    return (
        upload(rig.sibling, key, batch_file(marker(), 2)),
        download(rig.sibling, key, file_id),
        download(rig.sibling, key, file_id),
    )


@pytest.mark.parametrize("level", LEVELS)
def test_a_lowered_limit_reaches_the_other_proxy_process(rig: Rig, level: str) -> None:
    with rig.candidate.scenario() as scenario:
        team: Final = scenario.team(metadata={RECORDS: 3, DOWNLOADS: 5}) if level == "team" else None
        key: Final = _key(
            scenario, team, **({UPLOADS: ROOMY} if level == "team" else {UPLOADS: ROOMY, RECORDS: 3, DOWNLOADS: 5})
        )
        source: Final = IN_KEY if level == "key" else IN_TEAM
        assert _statuses(_lowered_probe(rig, key)) == [200, 200, 200]
        if team is None:
            rig.candidate.post("/key/update", {"key": key, "metadata": {UPLOADS: ROOMY, RECORDS: 1, DOWNLOADS: 1}})
        else:
            rig.candidate.post("/team/update", {"team_id": team, "metadata": {RECORDS: 1, DOWNLOADS: 1}})
        too_long, _, _ = eventually(
            partial(_lowered_probe, rig, key), lambda probe: _statuses(probe) == [413, 200, 429], seconds=30
        )
        assert_too_many_records(too_long, 1, source)


@asynccontextmanager
async def _async_clients(rig: Rig, count: int) -> AsyncIterator[tuple[httpx.AsyncClient, ...]]:
    async with (
        httpx.AsyncClient(base_url=rig.candidate.client.base_url, timeout=60, trust_env=False) as first,
        httpx.AsyncClient(base_url=rig.sibling.client.base_url, timeout=60, trust_env=False) as second,
    ):
        yield tuple((first, second)[index % 2] for index in range(count))


def _raise_then_lower(rig: Rig, key: str) -> None:
    rig.candidate.post("/key/update", {"key": key, "metadata": {DOWNLOADS: 8}})
    rig.candidate.post("/key/update", {"key": key, "metadata": {DOWNLOADS: 3}})


async def test_a_limit_changed_during_a_download_burst_only_answers_200_or_429(rig: Rig) -> None:
    with rig.candidate.scenario() as scenario:
        key: Final = _key(scenario, **{DOWNLOADS: 5})
        file_id: Final = f"file-{marker()}"
        headers: Final = {"Authorization": f"Bearer {key}"}
        minute_ends: Final = window_end(MINUTE_SECONDS, MINUTE_ROOM_SECONDS)
        async with _async_clients(rig, 30) as clients:
            *responses, _ = await asyncio.gather(
                *(client.get(f"/v1/files/{file_id}/content", headers=headers) for client in clients),
                asyncio.to_thread(_raise_then_lower, rig, key),
            )
        assert time.time() < minute_ends
        statuses: Final = [response.status_code for response in responses]
        assert set(statuses) <= {200, 429}, [response.text for response in responses]
        assert 3 <= statuses.count(200) <= 8, statuses
        assert len(downloads_seen(rig.provider.drain(), file_id)) == statuses.count(200)


async def test_concurrent_requests_across_processes_never_exceed_a_limit(rig: Rig) -> None:
    with rig.candidate.scenario() as scenario:
        key: Final = _key(scenario, **{UPLOADS: 5, DOWNLOADS: 4})
        mark: Final = marker()
        file_id: Final = f"file-{mark}"
        headers: Final = {"Authorization": f"Bearer {key}"}
        day_ends: Final = window_end(DAY_SECONDS, DAY_ROOM_SECONDS)
        minute_ends: Final = window_end(MINUTE_SECONDS, MINUTE_ROOM_SECONDS)
        async with _async_clients(rig, 24) as clients:
            uploads: Final = await asyncio.gather(
                *(
                    client.post(
                        "/v1/files",
                        data={"purpose": "batch"},
                        files={"file": ("batch.jsonl", batch_file(mark, 1), "application/jsonl")},
                        headers=headers,
                    )
                    for client in clients
                )
            )
            downloads: Final = await asyncio.gather(
                *(client.get(f"/v1/files/{file_id}/content", headers=headers) for client in clients[:20])
            )
        assert time.time() < min(day_ends, minute_ends)
        assert sorted(response.status_code for response in uploads) == [200] * 5 + [429] * 19
        assert sorted(response.status_code for response in downloads) == [200] * 4 + [429] * 16
        requests: Final = rig.provider.drain()
        assert len(uploads_seen(requests, mark)) == 5
        assert len(downloads_seen(requests, file_id)) == 4
        assert _counts(rig, UPLOADS, hashed(key)) == [5]
        assert _counts(rig, DOWNLOADS, f"{hashed(key)}:{file_id}:") == [4]
