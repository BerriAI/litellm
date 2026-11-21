import asyncio
import hashlib
import re
from typing import Final

import pytest

from litellm.integrations.lens.content import (
    MAX_CONTENT_READS,
    MAX_INLINE_RECORD_BYTES,
    ContentLimits,
    ContentReader,
    ContentRequest,
    Owner,
    ReadFailure,
    S3V2ContentStore,
    Stored,
    StoredReference,
    Unavailable,
    content_limits_from_settings,
    content_signing_key,
    content_store_from_settings,
    object_name,
    signed_reference,
)

_TOKEN: Final = "test-service-token-with-32-characters"
_OWNER: Final = Owner(team_id="team-a", api_key="hash-a", request_id="request-a")
_BODY: Final = '{"messages":"[\\"界\\"]","response":"done"}'


@pytest.mark.parametrize(
    "settings",
    (
        None,
        {},
        {"store": {"type": "lens"}},
        {"store": {"type": "lens", "content_storage": "gcs_bucket"}},
        {"store": "lens"},
    ),
    ids=("absent", "empty", "lens-default", "unsupported-storage", "malformed-store"),
)
def test_content_storage_and_limits_keep_defaults_unless_configured(settings: dict[str, object] | None) -> None:
    assert content_store_from_settings(settings) is None
    assert content_limits_from_settings(settings) == ContentLimits()


def test_configured_limits_are_capped_by_the_record_and_object_budgets() -> None:
    store: Final = {"content_storage": "s3_v2", "content_inline_bytes": 4096, "content_max_bytes": 1 << 40}
    assert content_store_from_settings({"store": store}) is not None
    limits: Final = content_limits_from_settings({"store": store})
    assert limits.inline_bytes == 4096
    assert limits.object_bytes == ContentLimits().object_bytes
    oversized: Final = content_limits_from_settings({"store": {"content_inline_bytes": MAX_INLINE_RECORD_BYTES + 1}})
    assert oversized.inline_bytes == MAX_INLINE_RECORD_BYTES


def test_object_names_bind_owner_and_content_without_embedding_caller_identifiers() -> None:
    owner: Final = Owner(team_id="team-a", api_key="hash-a", request_id="../team-b")
    name: Final = object_name(owner, 1_700_000_000_000, "a" * 64)
    assert name is not None and re.fullmatch(r"lens-content/2023-11-14/[0-9a-f]{64}\.json", name)
    assert name == object_name(owner, 1_700_000_000_000, "a" * 64)
    assert name != object_name(owner, 1_700_000_000_000, "b" * 64)
    assert name != object_name(
        Owner(team_id="team-b", api_key="hash-a", request_id="../team-b"), 1_700_000_000_000, "a" * 64
    )
    assert object_name(_OWNER, 10**20, "a" * 64) is None


def _reference(owner: Owner = _OWNER, body: str = _BODY) -> StoredReference:
    data: Final = body.encode()
    state: Final = Stored(object_key="logs/object.json", size=len(data), sha256=hashlib.sha256(data).hexdigest())
    return StoredReference.model_validate(signed_reference(content_signing_key(_TOKEN), owner, state))


class _Objects:
    def __init__(self, served: "asyncio.Future[str | None]") -> None:
        self.served: Final = served
        self.reads: Final = asyncio.Queue[str]()
        self.limits: Final = asyncio.Queue[int]()

    def object_key(self, name: str) -> str | None:
        return name

    def upload(self, key: str, body: str) -> "asyncio.Future[bool]":
        raise AssertionError("retrieval never uploads")

    def download(self, key: str, max_bytes: int) -> "asyncio.Future[str | None]":
        self.reads.put_nowait(key)
        self.limits.put_nowait(max_bytes)
        return self.served


def _served(value: str | None) -> "asyncio.Future[str | None]":
    future: Final = asyncio.get_running_loop().create_future()
    future.set_result(value)
    return future


def _request(owner: Owner = _OWNER, reference: StoredReference | None = None) -> ContentRequest:
    return ContentRequest(
        team_id=owner.team_id,
        api_key=owner.api_key,
        request_id=owner.request_id,
        reference=reference or _reference(),
    )


@pytest.mark.asyncio
async def test_authorized_reads_return_the_verified_object() -> None:
    objects: Final = _Objects(_served(_BODY))
    reader: Final = ContentReader(objects, _TOKEN)
    assert reader.authorized(f"Bearer {_TOKEN}")
    assert not reader.authorized("Bearer another-token-with-32-characters!!")
    assert await reader.read(_request()) == _BODY.encode()
    assert objects.reads.get_nowait() == "logs/object.json"
    assert objects.limits.get_nowait() == len(_BODY.encode())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "owner",
    (
        Owner(team_id="team-b", api_key="hash-a", request_id="request-a"),
        Owner(team_id="team-a", api_key="hash-b", request_id="request-a"),
        Owner(team_id="team-a", api_key="hash-a", request_id="request-b"),
    ),
    ids=("other-team", "other-key", "other-request"),
)
async def test_references_cannot_be_replayed_for_another_owner(owner: Owner) -> None:
    objects: Final = _Objects(_served(_BODY))
    assert await ContentReader(objects, _TOKEN).read(_request(owner, _reference())) is ReadFailure.NOT_FOUND
    assert objects.reads.empty()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("served", "failure"),
    (
        pytest.param('{"messages":"[\\"x\\"]","response":"done"}', ReadFailure.INTEGRITY, id="tampered"),
        pytest.param(_BODY + " ", ReadFailure.INTEGRITY, id="resized"),
        pytest.param(None, ReadFailure.UNAVAILABLE, id="storage-error"),
    ),
)
async def test_tampered_or_missing_objects_are_never_returned(served: str | None, failure: ReadFailure) -> None:
    assert await ContentReader(_Objects(_served(served)), _TOKEN).read(_request()) is failure


@pytest.mark.asyncio
async def test_saturated_reads_are_rejected_until_native_downloads_finish() -> None:
    served: Final = asyncio.get_running_loop().create_future()
    objects: Final = _Objects(served)
    reader: Final = ContentReader(objects, _TOKEN)
    readers: Final = tuple(asyncio.ensure_future(reader.read(_request())) for _ in range(MAX_CONTENT_READS))
    for _ in range(MAX_CONTENT_READS):
        await asyncio.wait_for(objects.reads.get(), timeout=5)
    try:
        assert await reader.read(_request()) is ReadFailure.UNAVAILABLE
        readers[0].cancel()
        assert await asyncio.gather(readers[0], return_exceptions=True) != [_BODY.encode()]
        assert await reader.read(_request()) is ReadFailure.UNAVAILABLE
    finally:
        released: Final = asyncio.Event()
        served.add_done_callback(lambda _: released.set())
        served.set_result(_BODY)
        await asyncio.wait_for(released.wait(), timeout=5)
    assert await asyncio.gather(*readers[1:]) == [_BODY.encode()] * (MAX_CONTENT_READS - 1)
    assert await reader.read(_request()) == _BODY.encode()


def test_signed_references_match_the_fixture_lens_verifies() -> None:
    key: Final = content_signing_key("fixture-service-token-with-32-chars!")
    owner: Final = Owner(team_id="team-\u00e9", api_key="hash-a", request_id="request:1")
    stored: Final = Stored(
        object_key="logs/lens-content/2023-11-14/abc.json",
        size=45,
        sha256="0e469198a46d22437fab286514c9c8a42c17552d50e8613225e2c31bd5407968",
    )
    assert signed_reference(key, owner, stored)["mac"] == (
        "7bfcd946ff46db7efbd3b916275e80a21558196308a27108c1f8004f24e26d0f"
    )
    assert signed_reference(key, owner, Unavailable.UPLOAD_FAILED)["mac"] == (
        "64836be49c032f404b9a140b82c67fe8fdbb5dc2483583fe5aee11e3926e9a12"
    )


class _BrokenObjects(_Objects):
    def download(self, key: str, max_bytes: int) -> "asyncio.Future[str | None]":
        raise RuntimeError("the native sink rejected the call")


@pytest.mark.asyncio
async def test_storage_errors_release_the_read_slot_and_report_unavailable() -> None:
    reader: Final = ContentReader(_BrokenObjects(_served(None)), _TOKEN)
    for _ in range(MAX_CONTENT_READS + 1):
        assert await reader.read(_request()) is ReadFailure.UNAVAILABLE
    failed: Final = asyncio.get_running_loop().create_future()
    failed.set_exception(RuntimeError("download failed"))
    assert await ContentReader(_Objects(failed), _TOKEN).read(_request()) is ReadFailure.UNAVAILABLE


@pytest.mark.asyncio
async def test_without_an_active_s3_v2_logger_the_store_names_and_moves_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm

    for name in ("callbacks", "success_callback", "_async_success_callback"):
        monkeypatch.setattr(litellm, name, [])
    for store in (S3V2ContentStore(), S3V2ContentStore(resolve=lambda: None)):
        assert store.object_key("lens-content/object.json") is None
        assert await store.upload("lens-content/object.json", "body") is False
        assert await store.download("lens-content/object.json", 4) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix", ("team-logs/", None), ids=("prefixed", "unprefixed"))
async def test_the_active_s3_v2_logger_stores_and_reads_content_within_the_reference_size(
    monkeypatch: pytest.MonkeyPatch, prefix: str | None
) -> None:
    from tests.unit.integrations.s3_v2_rust_support import rust_s3_logger, scripted_bucket

    with scripted_bucket() as bucket:
        logger: Final = rust_s3_logger(
            monkeypatch,
            s3_bucket_name="logs-bucket",
            s3_endpoint_url=bucket.url,
            s3_aws_access_key_id="AKIDEXAMPLE",
            s3_aws_secret_access_key="secret",
            s3_region_name="us-east-1",
            **({"s3_path": prefix} if prefix else {}),
        )
        try:
            store: Final = S3V2ContentStore(resolve=lambda: logger)
            key: Final = store.object_key("lens-content/object.json")
            assert key == ("team-logs/" if prefix else "") + "lens-content/object.json"
            assert key is not None
            assert await store.upload(key, _BODY) is True
            size: Final = len(_BODY.encode())
            assert await store.download(key, size) == _BODY
            assert await store.download(key, size - 1) is None
            bucket.reply("rejected", (403, "<Error><Code>AccessDenied</Code></Error>"))
            assert await store.upload(f"{key}.rejected", _BODY) is False
        finally:
            _ = logger.periodic_flush_task.cancel()
