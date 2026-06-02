from types import SimpleNamespace

import pytest

from litellm.exceptions import BadRequestError
from litellm.llms.vertex_ai.vector_stores.search_api.transformation import (
    VertexSearchAPIVectorStoreConfig,
)


def test_should_encode_vertex_search_vector_store_id_in_complete_url():
    config = VertexSearchAPIVectorStoreConfig()

    url = config.get_complete_url(
        api_base=None,
        litellm_params={
            "vertex_project": "test-project",
            "vertex_location": "global",
            "vertex_collection_id": "default/collection",
            "vector_store_id": "../../dataStores/other?x=1#frag",
        },
    )

    assert (
        url
        == "https://discoveryengine.googleapis.com/v1/projects/test-project/locations/global/collections/default%2Fcollection/dataStores/..%2F..%2FdataStores%2Fother%3Fx%3D1%23frag/servingConfigs/default_config"
    )


def test_should_reject_dot_segment_vertex_search_vector_store_id():
    config = VertexSearchAPIVectorStoreConfig()

    with pytest.raises(
        ValueError, match="vector_store_id cannot be a dot path segment"
    ):
        config.get_complete_url(
            api_base=None,
            litellm_params={
                "vertex_project": "test-project",
                "vertex_location": "global",
                "vector_store_id": "..",
            },
        )


def test_should_use_engines_url_when_engine_id_provided():
    config = VertexSearchAPIVectorStoreConfig()

    url = config.get_complete_url(
        api_base=None,
        litellm_params={
            "vertex_project": "test-project",
            "vertex_location": "global",
            "vertex_engine_id": "test-engine_1234",
        },
    )

    assert url == (
        "https://discoveryengine.googleapis.com/v1/"
        "projects/test-project/locations/global/"
        "collections/default_collection/engines/test-engine_1234/servingConfigs/default_serving_config"
    )


def test_engine_id_takes_precedence_over_vector_store_id():
    config = VertexSearchAPIVectorStoreConfig()

    url = config.get_complete_url(
        api_base=None,
        litellm_params={
            "vertex_project": "test-project",
            "vertex_location": "global",
            "vertex_engine_id": "test-engine_1234",
            "vector_store_id": "ignored-when-engine-set",
        },
    )

    assert "/engines/test-engine_1234/" in url
    assert "/dataStores/" not in url
    assert url.endswith("/servingConfigs/default_serving_config")


def test_should_encode_vertex_engine_id_in_complete_url():
    config = VertexSearchAPIVectorStoreConfig()

    url = config.get_complete_url(
        api_base=None,
        litellm_params={
            "vertex_project": "test-project",
            "vertex_location": "global",
            "vertex_engine_id": "../../engines/other?x=1#frag",
        },
    )

    assert url == (
        "https://discoveryengine.googleapis.com/v1/"
        "projects/test-project/locations/global/"
        "collections/default_collection/engines/..%2F..%2Fengines%2Fother%3Fx%3D1%23frag/servingConfigs/default_serving_config"
    )


def test_should_reject_dot_segment_vertex_engine_id():
    config = VertexSearchAPIVectorStoreConfig()

    with pytest.raises(
        ValueError, match="vertex_engine_id cannot be a dot path segment"
    ):
        config.get_complete_url(
            api_base=None,
            litellm_params={
                "vertex_project": "test-project",
                "vertex_location": "global",
                "vertex_engine_id": "..",
            },
        )


def test_should_raise_when_neither_engine_id_nor_vector_store_id_provided():
    config = VertexSearchAPIVectorStoreConfig()

    with pytest.raises(
        ValueError,
        match="vector_store_id is required when vertex_engine_id is not set",
    ):
        config.get_complete_url(
            api_base=None,
            litellm_params={
                "vertex_project": "test-project",
                "vertex_location": "global",
            },
        )


_ENGINE_BASE = (
    "https://discoveryengine.googleapis.com/v1/projects/p/locations/global/"
    "collections/default_collection/engines/app-2/servingConfigs/default_serving_config"
)

_DATASTORE_BASE = (
    "https://discoveryengine.googleapis.com/v1/projects/p/locations/global/"
    "collections/default_collection/dataStores/ds-1/servingConfigs/default_config"
)


def _search_request(**overrides):
    """Engine/app-mode search request (vertex_engine_id set)."""
    kwargs = dict(
        vector_store_id="vs",
        query="hello",
        vector_store_search_optional_params={},
        api_base=_ENGINE_BASE,
        litellm_logging_obj=SimpleNamespace(model_call_details={}),
        litellm_params={"vertex_engine_id": "app-2"},
    )
    kwargs.update(overrides)
    return VertexSearchAPIVectorStoreConfig().transform_search_vector_store_request(
        **kwargs
    )


def _datastore_search_request(**overrides):
    """Data-store-mode search request (no vertex_engine_id)."""
    kwargs = dict(
        vector_store_id="ds-1",
        query="hello",
        vector_store_search_optional_params={},
        api_base=_DATASTORE_BASE,
        litellm_logging_obj=SimpleNamespace(model_call_details={}),
        litellm_params={},
    )
    kwargs.update(overrides)
    return VertexSearchAPIVectorStoreConfig().transform_search_vector_store_request(
        **kwargs
    )


def test_search_request_defaults_to_query_and_pagesize_10():
    url, body = _search_request()

    assert url == _ENGINE_BASE + ":search"
    assert body == {"query": "hello", "pageSize": 10}


def test_search_request_maps_max_num_results_to_pagesize():
    _, body = _search_request(
        vector_store_search_optional_params={"max_num_results": 25}
    )

    assert body["pageSize"] == 25


def test_engine_search_request_forwards_datastorespecs():
    specs = [
        {
            "dataStore": "projects/p/locations/global/collections/default_collection/dataStores/ds-beta"
        }
    ]

    _, body = _search_request(extra_body={"dataStoreSpecs": specs})

    assert body["dataStoreSpecs"] == specs


def test_engine_search_request_forwards_num_results_per_data_store():
    _, body = _search_request(extra_body={"numResultsPerDataStore": 3})

    assert body["numResultsPerDataStore"] == 3


def test_datastore_search_request_rejects_datastorespecs():
    specs = [{"dataStore": "projects/p/.../dataStores/ds-beta"}]

    with pytest.raises(BadRequestError, match="data store mode"):
        _datastore_search_request(extra_body={"dataStoreSpecs": specs})


def test_datastore_search_request_rejects_num_results_per_data_store():
    with pytest.raises(BadRequestError, match="data store mode"):
        _datastore_search_request(extra_body={"numResultsPerDataStore": 3})


@pytest.mark.parametrize("field", ["branch", "servingConfig", "entity"])
def test_search_request_rejects_target_selecting_fields(field):
    with pytest.raises(BadRequestError, match="target-selecting"):
        _search_request(extra_body={field: "x"})


@pytest.mark.parametrize("field", ["branch", "servingConfig", "entity"])
def test_datastore_search_request_rejects_target_selecting_fields(field):
    with pytest.raises(BadRequestError, match="target-selecting"):
        _datastore_search_request(extra_body={field: "x"})


def test_search_request_rejects_unsupported_extra_body_field():
    with pytest.raises(BadRequestError, match="Unsupported Vertex AI Search extra_body"):
        _search_request(extra_body={"notARealField": True})


def test_rejected_extra_body_raises_http_400():
    with pytest.raises(BadRequestError) as exc_info:
        _search_request(extra_body={"notARealField": True})

    assert exc_info.value.status_code == 400


def test_search_request_forwards_supported_extra_body_fields():
    _, body = _search_request(
        extra_body={
            "filter": 'category: ANY("docs")',
            "boostSpec": {"conditionBoostSpecs": []},
        }
    )

    assert body["filter"] == 'category: ANY("docs")'
    assert body["boostSpec"] == {"conditionBoostSpecs": []}
    assert body["query"] == "hello"


def test_datastore_search_request_forwards_supported_extra_body_fields():
    _, body = _datastore_search_request(
        extra_body={"filter": 'category: ANY("docs")'}
    )

    assert body["filter"] == 'category: ANY("docs")'


def test_search_request_ignores_none_valued_extra_body_fields():
    _, body = _search_request(extra_body={"filter": None})

    assert "filter" not in body


def test_search_request_extra_body_takes_precedence_over_defaults():
    _, body = _search_request(
        vector_store_search_optional_params={"max_num_results": 5},
        extra_body={"pageSize": 50, "filter": 'category: ANY("docs")'},
    )

    assert body["pageSize"] == 50
    assert body["filter"] == 'category: ANY("docs")'


def test_search_request_joins_list_query():
    _, body = _search_request(query=["foo", "bar"])

    assert body["query"] == "foo bar"


def test_search_request_logs_effective_query_when_extra_body_overrides_query():
    log = SimpleNamespace(model_call_details={})

    _, body = _search_request(
        query="original",
        extra_body={"query": "from-extra-body"},
        litellm_logging_obj=log,
    )

    assert body["query"] == "from-extra-body"
    assert log.model_call_details["query"] == "from-extra-body"
