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
