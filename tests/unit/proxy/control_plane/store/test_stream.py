from collections.abc import Mapping
from typing import Final

import pytest
from pydantic import SecretStr

from litellm.proxy.control_plane.settings import DataplaneSpec
from litellm.proxy.control_plane.store.crypto import SecretCodec
from litellm.proxy.control_plane.store.models import Change, Entry
from litellm.proxy.control_plane.store.stream import RedisStream


class RedisRecorder:
    def __init__(self) -> None:
        self.fields: Mapping[str, str] = {}
        self.limit = 0
        self.approximate = False
        self.stream = ""
        self.after = ""
        self.block = 0
        self.closed = False

    async def xadd(self, name: str, fields: Mapping[str, str], *, maxlen: int, approximate: bool) -> object:
        self.fields = fields
        self.limit = maxlen
        self.approximate = approximate
        self.stream = name
        return "123-4"

    async def xread(self, streams: Mapping[str, str], *, count: int, block: int) -> object:
        self.after = streams[self.stream]
        self.block = block
        return [[self.stream, [["123-4", self.fields]]]]

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_stream_preserves_encrypted_entities_and_actor_and_uses_bounded_blocking_reads_and_approximate_trim() -> (
    None
):
    client: Final = RedisRecorder()
    stream: Final = RedisStream(client, "cp:config", 100)
    spec: Final = DataplaneSpec.model_validate(
        {"url": "https://runtime.test", "service_token": {"token": "plaintext-secret"}}
    )
    codec: Final = SecretCodec(SecretStr("configured-key"))
    change: Final = Change(op="upsert", kind="dataplane", name="live", entity=codec.encode(spec), actor="operator")
    revision: Final = await stream.append(change)
    assert revision == "123-4"
    assert (client.stream, client.limit, client.approximate) == ("cp:config", 100, True)
    assert "plaintext-secret" not in str(client.fields)
    records: Final = await stream.read("123-3")
    assert records == (Entry(revision=revision, change=change),)
    assert client.after == "123-3" and client.block > 0
    assert codec.decode(records[0].change.entity) == spec
    await stream.close()
    assert client.closed
