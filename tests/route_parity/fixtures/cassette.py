from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from itertools import accumulate
from typing import Final, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, TypeAdapter
from vcr.serialize import serialize
from vcr.serializers import yamlserializer

from tests.route_parity.fixtures.recording import RecordedInteraction
from tests.route_parity.recorded_http import (
    HttpHeader,
    RecordedHttpResponse,
    RecordedHttpStreamResponse,
    RecordedResponse,
    RecordedStreamChunk,
)

_OBJECT: Final = TypeAdapter(dict[str, object])


class _CassetteModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)


class _Body(_CassetteModel):
    string: str | bytes

    def as_bytes(self) -> bytes:
        return self.string.encode("utf-8") if isinstance(self.string, str) else self.string


class _Status(_CassetteModel):
    code: int
    message: str


class _Request(_CassetteModel):
    method: str
    uri: str
    body: str | bytes | None
    headers: dict[str, tuple[str, ...]]


class _Response(_CassetteModel):
    status: _Status
    headers: dict[str, tuple[str, ...]]
    body: _Body
    chunk_lengths: tuple[int, ...] | None = Field(default=None, alias="x-litellm-chunk-lengths")

    def recorded_response(self) -> RecordedResponse:
        headers: Final = tuple(
            HttpHeader(name=name, value=value) for name, values in self.headers.items() for value in values
        )
        body: Final = self.body.as_bytes()
        if self.chunk_lengths is None:
            return RecordedHttpResponse.from_bytes(self.status.code, headers, body)
        if any(length < 0 for length in self.chunk_lengths) or sum(self.chunk_lengths) != len(body):
            raise ValueError("cassette stream chunk lengths do not match the response body")
        offsets: Final = tuple(accumulate(self.chunk_lengths, initial=0))
        return RecordedHttpStreamResponse(
            kind="http_stream",
            status_code=self.status.code,
            headers=headers,
            chunks=tuple(RecordedStreamChunk.from_bytes(body[start:end]) for start, end in zip(offsets, offsets[1:])),
        )


class _Interaction(_CassetteModel):
    request: _Request
    response: _Response


class _ParityMetadata(_CassetteModel):
    schema_version: Literal[1]
    request_source: Literal["recorded", "python_replay"]
    case: dict[str, object]


class ParityCassette(_CassetteModel):
    version: Literal[1]
    recorded_at: AwareDatetime
    ttl_seconds: Literal[0]
    interactions: tuple[_Interaction, ...]
    parity: _ParityMetadata = Field(alias="x-litellm")

    def case_data(self) -> dict[str, object]:
        return {
            **self.parity.case,
            "provider_responses": tuple(item.response.recorded_response() for item in self.interactions),
        }


def _response_dict(response: RecordedResponse) -> dict[str, object]:
    headers: Final = {
        name: [header.value for header in response.headers if header.name == name]
        for name in dict.fromkeys(header.name for header in response.headers)
    }
    chunks: Final = (
        tuple(chunk.data_bytes() for chunk in response.chunks)
        if isinstance(response, RecordedHttpStreamResponse)
        else None
    )
    body: Final = response.body_bytes() if isinstance(response, RecordedHttpResponse) else b"".join(chunks or ())
    return {
        "status": {"code": response.status_code, "message": ""},
        "headers": headers,
        "body": {"string": body},
        **({"x-litellm-chunk-lengths": list(map(len, chunks))} if chunks is not None else {}),
    }


def serialize_cassette(
    case: Mapping[str, object],
    interactions: tuple[RecordedInteraction, ...],
    recorded_at: datetime,
    request_source: Literal["recorded", "python_replay"],
) -> str:
    normalized: Final = _OBJECT.validate_python(
        yamlserializer.deserialize(
            serialize(
                {
                    "requests": [item.request for item in interactions],
                    "responses": [_response_dict(item.response) for item in interactions],
                },
                yamlserializer,
            )
        )
    )
    payload: Final = {
        **normalized,
        "recorded_at": recorded_at.isoformat(),
        "ttl_seconds": 0,
        "x-litellm": {
            "schema_version": 1,
            "request_source": request_source,
            "case": {key: value for key, value in case.items() if key != "provider_responses"},
        },
    }
    ParityCassette.model_validate(payload).case_data()
    return str(yamlserializer.serialize(payload))


def deserialize_cassette(contents: str) -> ParityCassette:
    return ParityCassette.model_validate(yamlserializer.deserialize(contents))
