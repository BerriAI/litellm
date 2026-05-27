from unittest.mock import MagicMock

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


def test_datastore_mode_requires_vector_store_id():
    """Datastore-mode URL still requires vector_store_id (backwards compat)."""
    config = VertexSearchAPIVectorStoreConfig()
    with pytest.raises(ValueError, match="vector_store_id is required"):
        config.get_complete_url(
            api_base=None,
            litellm_params={
                "vertex_project": "test-project",
                "vertex_location": "global",
            },
        )


def test_engine_mode_emits_app_level_url():
    """vertex_engine_id selects engines/{id} URL instead of dataStores/{id}."""
    config = VertexSearchAPIVectorStoreConfig()

    url = config.get_complete_url(
        api_base=None,
        litellm_params={
            "vertex_project": "test-project",
            "vertex_location": "global",
            "vertex_engine_id": "my-app_1234567890",
        },
    )

    assert url == (
        "https://discoveryengine.googleapis.com/v1/"
        "projects/test-project/locations/global/"
        "collections/default_collection/engines/my-app_1234567890/"
        "servingConfigs/default_config"
    )


def test_engine_mode_accepts_vertex_app_id_alias():
    """vertex_app_id is accepted as an alias for vertex_engine_id."""
    config = VertexSearchAPIVectorStoreConfig()

    url = config.get_complete_url(
        api_base=None,
        litellm_params={
            "vertex_project": "test-project",
            "vertex_location": "us-central1",
            "vertex_app_id": "my-app",
        },
    )

    assert "engines/my-app" in url
    assert "dataStores/" not in url
    assert url.endswith("/servingConfigs/default_config")


def test_engine_mode_does_not_require_vector_store_id():
    """vertex_engine_id alone is enough; datastore id is optional in this mode."""
    config = VertexSearchAPIVectorStoreConfig()
    # Should not raise.
    url = config.get_complete_url(
        api_base=None,
        litellm_params={
            "vertex_project": "p",
            "vertex_location": "global",
            "vertex_engine_id": "engine-1",
        },
    )
    assert "engines/engine-1/" in url


def test_engine_id_is_url_path_encoded():
    """Malicious engine ids cannot escape into adjacent path segments."""
    config = VertexSearchAPIVectorStoreConfig()
    url = config.get_complete_url(
        api_base=None,
        litellm_params={
            "vertex_project": "p",
            "vertex_location": "global",
            "vertex_engine_id": "../dataStores/other",
        },
    )
    assert "engines/..%2FdataStores%2Fother/" in url


def test_custom_serving_config_id_used_in_url():
    """vertex_serving_config_id overrides the trailing serving-config segment."""
    config = VertexSearchAPIVectorStoreConfig()

    url = config.get_complete_url(
        api_base=None,
        litellm_params={
            "vertex_project": "p",
            "vertex_location": "global",
            "vertex_engine_id": "engine-1",
            "vertex_serving_config_id": "default_search",
        },
    )
    assert url.endswith("/servingConfigs/default_search")


def test_serving_config_id_is_url_path_encoded():
    config = VertexSearchAPIVectorStoreConfig()
    url = config.get_complete_url(
        api_base=None,
        litellm_params={
            "vertex_project": "p",
            "vertex_location": "global",
            "vector_store_id": "ds-1",
            "vertex_serving_config_id": "weird/value",
        },
    )
    assert url.endswith("/servingConfigs/weird%2Fvalue")


def _logger():
    logger = MagicMock()
    logger.model_call_details = {}
    return logger


def test_transform_search_attaches_data_store_specs():
    """vertex_data_store_specs is forwarded as dataStoreSpecs in the body."""
    config = VertexSearchAPIVectorStoreConfig()
    specs = [
        {
            "dataStore": (
                "projects/123/locations/global/collections/default_collection/"
                "dataStores/ds-a"
            )
        },
        {
            "dataStore": (
                "projects/123/locations/global/collections/default_collection/"
                "dataStores/ds-b"
            ),
            "filter": "category: ANY(\"docs\")",
        },
    ]
    url, body = config.transform_search_vector_store_request(
        vector_store_id="unused-in-engine-mode",
        query="hello",
        vector_store_search_optional_params={},
        api_base="https://discoveryengine.googleapis.com/v1/.../servingConfigs/default_config",
        litellm_logging_obj=_logger(),
        litellm_params={"vertex_data_store_specs": specs},
    )

    assert url.endswith(":search")
    assert body["query"] == "hello"
    assert body["pageSize"] == 10
    assert body["dataStoreSpecs"] == specs


def test_transform_search_rejects_non_list_data_store_specs():
    config = VertexSearchAPIVectorStoreConfig()
    with pytest.raises(ValueError, match="vertex_data_store_specs must be a list"):
        config.transform_search_vector_store_request(
            vector_store_id="ds",
            query="q",
            vector_store_search_optional_params={},
            api_base="https://example/x",
            litellm_logging_obj=_logger(),
            litellm_params={"vertex_data_store_specs": {"dataStore": "x"}},
        )


def test_transform_search_merges_extra_body():
    """extra_body merges into the request body and can override defaults."""
    config = VertexSearchAPIVectorStoreConfig()
    url, body = config.transform_search_vector_store_request(
        vector_store_id="ds-1",
        query="hello",
        vector_store_search_optional_params={},
        api_base="https://example/x",
        litellm_logging_obj=_logger(),
        litellm_params={},
        extra_body={"pageSize": 25, "numResultsPerDataStore": 5},
    )
    assert body["query"] == "hello"
    assert body["pageSize"] == 25
    assert body["numResultsPerDataStore"] == 5


def test_extra_body_data_store_specs_wins_over_litellm_param():
    """Both knobs set -> extra_body['dataStoreSpecs'] wins."""
    config = VertexSearchAPIVectorStoreConfig()
    from_param = [{"dataStore": "from-param"}]
    from_body = [{"dataStore": "from-body"}]
    _, body = config.transform_search_vector_store_request(
        vector_store_id="ds-1",
        query="q",
        vector_store_search_optional_params={},
        api_base="https://example/x",
        litellm_logging_obj=_logger(),
        litellm_params={"vertex_data_store_specs": from_param},
        extra_body={"dataStoreSpecs": from_body},
    )
    assert body["dataStoreSpecs"] == from_body


def test_transform_search_query_list_is_space_joined():
    config = VertexSearchAPIVectorStoreConfig()
    _, body = config.transform_search_vector_store_request(
        vector_store_id="ds-1",
        query=["hello", "world"],
        vector_store_search_optional_params={},
        api_base="https://example/x",
        litellm_logging_obj=_logger(),
        litellm_params={},
    )
    assert body["query"] == "hello world"


def test_transform_search_rejects_non_dict_data_store_spec_entry():
    """Each DataStoreSpec list entry must be a dict (addresses Greptile review)."""
    config = VertexSearchAPIVectorStoreConfig()
    with pytest.raises(
        ValueError,
        match=r"vertex_data_store_specs\[1\] must be a DataStoreSpec dict",
    ):
        config.transform_search_vector_store_request(
            vector_store_id="ds",
            query="q",
            vector_store_search_optional_params={},
            api_base="https://example/x",
            litellm_logging_obj=_logger(),
            litellm_params={
                "vertex_data_store_specs": [
                    {"dataStore": "ok"},
                    "not-a-dict",
                ]
            },
        )
