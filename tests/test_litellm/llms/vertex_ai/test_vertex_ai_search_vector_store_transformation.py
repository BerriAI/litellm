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
