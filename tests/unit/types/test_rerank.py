import json
from typing import Final

import pytest
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, ConfigDict

from litellm.types.rerank import RerankResponse


def test_rerank_response_keeps_result_order_and_null_meta_fields() -> None:
    payload: Final = {
        "id": "ab0fcca0-b617-11ef-b292-0242ac110002",
        "results": [
            {"index": 2, "relevance_score": 0.9958819150924683},
            {"index": 0, "relevance_score": 0.001293411129154265},
            {"index": 1, "relevance_score": 7.641685078851879e-05},
            {"index": 3, "relevance_score": 7.621097756782547e-05},
        ],
        "meta": {"api_version": None, "billed_units": None, "tokens": None},
    }

    assert RerankResponse(**payload).model_dump() == payload


def test_rerank_response_round_trips_the_provider_usage_block_with_unknown_keys() -> None:
    payload: Final = {
        "id": "rerank-1",
        "results": [{"index": 0, "relevance_score": 0.5}],
        "meta": {"billed_units": {"search_units": 1}, "tokens": {}},
        "usage": {"search_units": 1, "cost": 0.002, "is_byok": False, "cost_details": {"upstream": 0.002}},
    }

    response: Final = RerankResponse(**payload)

    assert response.model_dump(by_alias=True) == payload
    for dumped in (response.model_dump_json(), response.model_dump_json(by_alias=True)):
        assert RerankResponse.model_validate_json(dumped).model_dump(by_alias=True) == payload


class _RerankEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True)

    response: RerankResponse


def _serialized_forms(response: RerankResponse) -> tuple[object, ...]:
    envelope: Final = _RerankEnvelope(response=response)
    return (
        response.model_dump(),
        response.model_dump(by_alias=True),
        response.model_dump(mode="json", by_alias=True),
        json.loads(response.model_dump_json()),
        json.loads(response.model_dump_json(by_alias=True)),
        envelope.model_dump()["response"],
        json.loads(envelope.model_dump_json(by_alias=True))["response"],
        jsonable_encoder(response),
        jsonable_encoder({"response": response})["response"],
    )


def test_rerank_response_without_provider_usage_serializes_no_usage_key_in_any_form() -> None:
    payload: Final = {"id": "rerank-1", "results": [{"index": 0, "relevance_score": 0.5}], "meta": None}

    assert all(form == payload for form in _serialized_forms(RerankResponse(**payload)))


def test_rerank_response_client_json_carries_the_provider_usage_in_nested_and_fastapi_forms() -> None:
    usage: Final = {"search_units": 1, "cost": 0.002, "vendor_detail": {"tier": "x"}}
    response: Final = RerankResponse(id="rerank-1", results=[], meta=None, usage=usage)
    envelope: Final = _RerankEnvelope(response=response)

    assert jsonable_encoder(response)["usage"] == usage
    assert jsonable_encoder({"response": response})["response"]["usage"] == usage
    assert json.loads(envelope.model_dump_json(by_alias=True))["response"]["usage"] == usage


@pytest.mark.parametrize("malformed_usage", ["n/a", ["search_units", 1], 7, True])
def test_rerank_response_accepts_a_malformed_provider_usage_by_dropping_it(malformed_usage: object) -> None:
    payload: Final = {"id": "rerank-1", "results": [], "meta": None}

    response: Final = RerankResponse.model_validate({**payload, "usage": malformed_usage})

    assert response.provider_usage is None
    assert response.model_dump(by_alias=True) == payload
