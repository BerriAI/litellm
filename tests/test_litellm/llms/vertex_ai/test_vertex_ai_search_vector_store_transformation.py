from types import SimpleNamespace

import pytest

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


def _search_request(**overrides):
    kwargs = dict(
        vector_store_id="vs",
        query="hello",
        vector_store_search_optional_params={},
        api_base=_ENGINE_BASE,
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


def test_search_request_passes_datastorespecs_through_extra_body():
    specs = [
        {
            "dataStore": "projects/p/locations/global/collections/default_collection/dataStores/ds-beta"
        }
    ]

    _, body = _search_request(extra_body={"dataStoreSpecs": specs})

    assert body["dataStoreSpecs"] == specs
    assert body["query"] == "hello"
    assert body["pageSize"] == 10


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
