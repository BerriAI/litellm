import pytest

from litellm.llms.vertex_ai.vector_stores.search_api.transformation import (
    VertexSearchAPIVectorStoreConfig,
)


# Datastore path (existing behaviour).
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


def test_should_require_vector_store_id_or_app_id():
    config = VertexSearchAPIVectorStoreConfig()

    with pytest.raises(ValueError, match="vector_store_id is required"):
        config.get_complete_url(
            api_base=None,
            litellm_params={
                "vertex_project": "test-project",
                "vertex_location": "global",
            },
        )


def test_datastore_path_honours_custom_serving_config():
    config = VertexSearchAPIVectorStoreConfig()

    url = config.get_complete_url(
        api_base=None,
        litellm_params={
            "vertex_project": "p",
            "vertex_location": "global",
            "vector_store_id": "ds1",
            "vertex_serving_config": "my_search_config",
        },
    )

    assert url.endswith("dataStores/ds1/servingConfigs/my_search_config")


# Engine / app-level path (LIT-3275).
def test_app_level_url_uses_engines_path_with_default_serving_config():
    config = VertexSearchAPIVectorStoreConfig()

    url = config.get_complete_url(
        api_base=None,
        litellm_params={
            "vertex_project": "test-project",
            "vertex_location": "global",
            "vertex_collection_id": "default_collection",
            "vertex_app_id": "my-app",
        },
    )

    assert url == (
        "https://discoveryengine.googleapis.com/v1/"
        "projects/test-project/locations/global/"
        "collections/default_collection/engines/my-app/"
        "servingConfigs/default_search"
    )


def test_vertex_engine_id_is_accepted_as_alias_for_vertex_app_id():
    config = VertexSearchAPIVectorStoreConfig()

    url = config.get_complete_url(
        api_base=None,
        litellm_params={
            "vertex_project": "p",
            "vertex_location": "us-central1",
            "vertex_engine_id": "engine-42",
        },
    )

    assert (
        "/locations/us-central1/collections/default_collection/engines/engine-42/"
        in url
    )


def test_app_id_path_does_not_require_vector_store_id():
    config = VertexSearchAPIVectorStoreConfig()

    url = config.get_complete_url(
        api_base=None,
        litellm_params={
            "vertex_project": "p",
            "vertex_location": "global",
            "vertex_app_id": "my-app",
        },
    )

    assert "/engines/my-app/" in url
    assert "/dataStores/" not in url


def test_app_id_path_takes_precedence_over_vector_store_id():
    config = VertexSearchAPIVectorStoreConfig()

    url = config.get_complete_url(
        api_base=None,
        litellm_params={
            "vertex_project": "p",
            "vertex_location": "global",
            "vertex_app_id": "primary-app",
            "vector_store_id": "fallback-ds",
        },
    )

    assert "/engines/primary-app/" in url
    assert "/dataStores/" not in url


def test_app_id_path_encodes_unsafe_characters():
    config = VertexSearchAPIVectorStoreConfig()

    url = config.get_complete_url(
        api_base=None,
        litellm_params={
            "vertex_project": "p",
            "vertex_location": "global",
            "vertex_app_id": "../escape/attempt",
            "vertex_collection_id": "default/collection",
        },
    )

    assert "engines/..%2Fescape%2Fattempt" in url
    assert "collections/default%2Fcollection" in url


def test_app_id_path_rejects_dot_segment_app_id():
    config = VertexSearchAPIVectorStoreConfig()

    with pytest.raises(ValueError, match="vertex_app_id cannot be a dot path segment"):
        config.get_complete_url(
            api_base=None,
            litellm_params={
                "vertex_project": "p",
                "vertex_location": "global",
                "vertex_app_id": "..",
            },
        )


def test_app_id_path_honours_custom_serving_config():
    config = VertexSearchAPIVectorStoreConfig()

    url = config.get_complete_url(
        api_base=None,
        litellm_params={
            "vertex_project": "p",
            "vertex_location": "global",
            "vertex_app_id": "my-app",
            "vertex_serving_config": "alt_search",
        },
    )

    assert url.endswith("engines/my-app/servingConfigs/alt_search")


class _LoggingStub:
    def __init__(self) -> None:
        self.model_call_details: dict = {}


def test_transform_request_app_level_with_vertex_datastores_bare_ids():
    config = VertexSearchAPIVectorStoreConfig()
    logging_obj = _LoggingStub()

    url, body = config.transform_search_vector_store_request(
        vector_store_id="ignored",
        query="hello",
        vector_store_search_optional_params={},
        api_base=(
            "https://discoveryengine.googleapis.com/v1/projects/p/locations/global/"
            "collections/default_collection/engines/my-app/servingConfigs/default_search"
        ),
        litellm_logging_obj=logging_obj,
        litellm_params={
            "vertex_project": "p",
            "vertex_location": "global",
            "vertex_app_id": "my-app",
            "vertex_datastores": ["ds-alpha", "ds-beta"],
        },
    )

    assert url.endswith(":search")
    assert body["query"] == "hello"
    assert body["dataStoreSpecs"] == [
        {
            "dataStore": "projects/p/locations/global/collections/default_collection/dataStores/ds-alpha"
        },
        {
            "dataStore": "projects/p/locations/global/collections/default_collection/dataStores/ds-beta"
        },
    ]


def test_transform_request_app_level_with_vertex_datastores_full_paths():
    config = VertexSearchAPIVectorStoreConfig()
    logging_obj = _LoggingStub()

    full_path = (
        "projects/other-proj/locations/us/collections/default_collection/"
        "dataStores/ds-x"
    )
    _, body = config.transform_search_vector_store_request(
        vector_store_id="ignored",
        query="hi",
        vector_store_search_optional_params={},
        api_base="https://discoveryengine.googleapis.com/v1/projects/p/locations/global/collections/default_collection/engines/my-app/servingConfigs/default_search",
        litellm_logging_obj=logging_obj,
        litellm_params={
            "vertex_project": "p",
            "vertex_location": "global",
            "vertex_app_id": "my-app",
            "vertex_datastores": [full_path],
        },
    )

    assert body["dataStoreSpecs"] == [{"dataStore": full_path}]


def test_transform_request_app_level_without_vertex_datastores_omits_specs():
    config = VertexSearchAPIVectorStoreConfig()
    logging_obj = _LoggingStub()

    _, body = config.transform_search_vector_store_request(
        vector_store_id="ignored",
        query="hi",
        vector_store_search_optional_params={},
        api_base="https://discoveryengine.googleapis.com/v1/projects/p/locations/global/collections/default_collection/engines/my-app/servingConfigs/default_search",
        litellm_logging_obj=logging_obj,
        litellm_params={
            "vertex_project": "p",
            "vertex_location": "global",
            "vertex_app_id": "my-app",
        },
    )

    assert "dataStoreSpecs" not in body


def test_transform_request_datastore_path_never_emits_specs():
    config = VertexSearchAPIVectorStoreConfig()
    logging_obj = _LoggingStub()

    _, body = config.transform_search_vector_store_request(
        vector_store_id="ds-1",
        query="hi",
        vector_store_search_optional_params={},
        api_base="https://discoveryengine.googleapis.com/v1/projects/p/locations/global/collections/default_collection/dataStores/ds-1/servingConfigs/default_config",
        litellm_logging_obj=logging_obj,
        litellm_params={
            "vertex_project": "p",
            "vertex_location": "global",
            "vector_store_id": "ds-1",
            "vertex_datastores": ["something"],
        },
    )

    assert "dataStoreSpecs" not in body


def test_transform_request_rejects_non_list_vertex_datastores():
    config = VertexSearchAPIVectorStoreConfig()
    logging_obj = _LoggingStub()

    with pytest.raises(ValueError, match="vertex_datastores must be a list"):
        config.transform_search_vector_store_request(
            vector_store_id="ignored",
            query="hi",
            vector_store_search_optional_params={},
            api_base="https://example.com",
            litellm_logging_obj=logging_obj,
            litellm_params={
                "vertex_project": "p",
                "vertex_location": "global",
                "vertex_app_id": "my-app",
                "vertex_datastores": "ds-alpha",
            },
        )


def test_full_call_sequence_matches_proxy_call_shape():
    """Regression test for the proxy's actual call shape:

    ``litellm/llms/custom_httpx/llm_http_handler.py`` calls
    ``get_complete_url(litellm_params=dict(litellm_params))`` and then
    ``transform_search_vector_store_request(litellm_params=dict(litellm_params))``
    — i.e. each provider method gets its OWN fresh shallow copy of
    ``litellm_params``. Lock in that the engine path produces the
    expected URL + dataStoreSpecs end-to-end under that contract.
    """
    config = VertexSearchAPIVectorStoreConfig()
    logging_obj = _LoggingStub()

    base_params = {
        "vertex_project": "demo-proj",
        "vertex_location": "global",
        "vertex_collection_id": "default_collection",
        "vertex_app_id": "my-app",
        "vertex_datastores": ["ds-alpha", "ds-beta"],
    }

    # Step 1: get_complete_url with its own dict copy.
    url = config.get_complete_url(api_base=None, litellm_params=dict(base_params))
    assert url == (
        "https://discoveryengine.googleapis.com/v1/"
        "projects/demo-proj/locations/global/"
        "collections/default_collection/engines/my-app/"
        "servingConfigs/default_search"
    )

    # Step 2: transform_search with its own (fresh) dict copy.
    request_url, body = config.transform_search_vector_store_request(
        vector_store_id="ignored",
        query="capital of France",
        vector_store_search_optional_params={},
        api_base=url,
        litellm_logging_obj=logging_obj,
        litellm_params=dict(base_params),
    )

    assert request_url == url + ":search"
    assert body["query"] == "capital of France"
    assert body["dataStoreSpecs"] == [
        {
            "dataStore": "projects/demo-proj/locations/global/collections/default_collection/dataStores/ds-alpha"
        },
        {
            "dataStore": "projects/demo-proj/locations/global/collections/default_collection/dataStores/ds-beta"
        },
    ]
