import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest

from litellm.llms.mongodb.common_utils import (
    MongoClientKey,
    get_async_client,
    get_sync_client,
    reset_client_cache,
    translate_mongo_error,
)
from litellm.llms.mongodb.vector_stores.transformation import (
    MongoDBVectorStoreConfig,
    _MongoDBSearchParams,
)
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

CONNECTION_STRING = "mongodb+srv://user:pw@cluster.example.mongodb.net"
INDEX = "movies_vector_index"

BASE_PARAMS = {
    "litellm_embedding_model": "openai/text-embedding-ada-002",
    "mongodb_connection_string": CONNECTION_STRING,
    "mongodb_database": "sample_mflix",
    "mongodb_collection": "embedded_movies",
}


class FakeCollection:
    def __init__(self, documents, error=None):
        self.documents = documents
        self.error = error
        self.pipeline = None

    def aggregate(self, pipeline):
        self.pipeline = pipeline
        if self.error is not None:
            raise self.error
        return iter(self.documents)


class FakeAsyncCollection(FakeCollection):
    async def aggregate(self, pipeline):
        self.pipeline = pipeline
        if self.error is not None:
            raise self.error

        async def cursor():
            for document in self.documents:
                yield document

        return cursor()


class FakeDatabase:
    def __init__(self, collection):
        self.collection = collection
        self.requested_collection = None

    def __getitem__(self, name):
        self.requested_collection = name
        return self.collection


class FakeClient:
    def __init__(self, collection):
        self.database = FakeDatabase(collection)
        self.requested_database = None

    def __getitem__(self, name):
        self.requested_database = name
        return self.database


class FakeEmbeddingFn:
    def __init__(self, embedding):
        self.embedding = embedding
        self.captured_kwargs = None

    def __call__(self, **kwargs):
        self.captured_kwargs = kwargs
        return SimpleNamespace(data=[{"embedding": self.embedding}] if self.embedding is not None else [])


class FakeAsyncEmbeddingFn(FakeEmbeddingFn):
    async def __call__(self, **kwargs):
        self.captured_kwargs = kwargs
        return SimpleNamespace(data=[{"embedding": self.embedding}] if self.embedding is not None else [])


def _config(documents=(), embedding=(0.1, 0.2, 0.3), error=None):
    collection = FakeCollection(list(documents), error)
    client = FakeClient(collection)
    config = MongoDBVectorStoreConfig(
        embedding_fn=FakeEmbeddingFn(list(embedding) if embedding is not None else None),
        sync_client_factory=lambda key: client,
    )
    return config, client, collection


def _async_config(documents=(), embedding=(0.1, 0.2, 0.3), error=None):
    collection = FakeAsyncCollection(list(documents), error)
    client = FakeClient(collection)
    config = MongoDBVectorStoreConfig(
        aembedding_fn=FakeAsyncEmbeddingFn(list(embedding) if embedding is not None else None),
        async_client_factory=lambda key: client,
    )
    return config, client, collection


def _search(config, query="a lone astronaut", optional_params=None, litellm_params=None, timeout=None):
    return config.execute_search_vector_store_request(
        vector_store_id=INDEX,
        query=query,
        vector_store_search_optional_params=optional_params or {},
        litellm_logging_obj=MagicMock(),
        litellm_params={**BASE_PARAMS, **(litellm_params or {})},
        timeout=timeout,
    )


async def _asearch(config, query="a lone astronaut", optional_params=None, litellm_params=None):
    return await config.aexecute_search_vector_store_request(
        vector_store_id=INDEX,
        query=query,
        vector_store_search_optional_params=optional_params or {},
        litellm_logging_obj=MagicMock(),
        litellm_params={**BASE_PARAMS, **(litellm_params or {})},
    )


def _stage(collection, name):
    return next(stage[name] for stage in collection.pipeline if name in stage)


def test_search_builds_vector_search_stage_against_the_named_index():
    config, client, collection = _config()

    _search(config, optional_params={"max_num_results": 5})

    assert client.requested_database == "sample_mflix"
    assert client.database.requested_collection == "embedded_movies"
    assert _stage(collection, "$vectorSearch") == {
        "index": INDEX,
        "path": "embedding",
        "queryVector": [0.1, 0.2, 0.3],
        "numCandidates": 100,
        "limit": 5,
    }


def test_search_projects_the_text_field_and_the_similarity_score():
    config, _, collection = _config()

    _search(config)

    assert _stage(collection, "$project") == {"text": 1, "score": {"$meta": "vectorSearchScore"}}


def test_search_defaults_to_ten_results():
    config, _, collection = _config()

    _search(config)

    assert _stage(collection, "$vectorSearch")["limit"] == 10


def test_search_honors_custom_field_names():
    config, _, collection = _config()

    _search(
        config,
        litellm_params={"mongodb_embedding_field": "plot_embedding", "mongodb_text_field": "plot"},
    )

    assert _stage(collection, "$vectorSearch")["path"] == "plot_embedding"
    assert _stage(collection, "$project") == {"plot": 1, "score": {"$meta": "vectorSearchScore"}}


def test_num_candidates_scales_with_the_requested_limit():
    config, _, collection = _config()

    _search(config, optional_params={"max_num_results": 40})

    assert _stage(collection, "$vectorSearch")["numCandidates"] == 400


def test_num_candidates_can_be_overridden():
    config, _, collection = _config()

    _search(config, optional_params={"max_num_results": 5}, litellm_params={"mongodb_num_candidates": 250})

    assert _stage(collection, "$vectorSearch")["numCandidates"] == 250


@pytest.mark.parametrize("configured", [4, 10_001])
def test_num_candidates_below_the_limit_or_above_the_ceiling_is_rejected(configured):
    config, _, _ = _config()

    with pytest.raises(ValueError, match="mongodb_num_candidates"):
        _search(config, optional_params={"max_num_results": 5}, litellm_params={"mongodb_num_candidates": configured})


def test_list_query_is_joined_into_one_embedding_input():
    config, _, _ = _config()
    embedding_fn = config.embedding_fn

    _search(config, query=["deep", "space", "rescue"])

    assert embedding_fn.captured_kwargs["input"] == ["deep space rescue"]


def test_embedding_config_is_expanded_into_the_embedding_call():
    config, _, _ = _config()
    embedding_fn = config.embedding_fn

    _search(config, litellm_params={"litellm_embedding_config": {"api_base": "https://example.test", "timeout": 7}})

    assert embedding_fn.captured_kwargs["api_base"] == "https://example.test"
    assert embedding_fn.captured_kwargs["timeout"] == 7
    assert embedding_fn.captured_kwargs["model"] == "openai/text-embedding-ada-002"


def test_response_maps_documents_to_openai_shaped_results():
    documents = [
        {"_id": "abc123", "text": "an astronaut adrift", "score": 0.94},
        {"_id": "def456", "text": "a robot dog", "score": 0.81},
    ]
    config, _, _ = _config(documents=documents)

    response = _search(config)

    assert response["object"] == "vector_store.search_results.page"
    assert response["search_query"] == "a lone astronaut"
    assert [result["score"] for result in response["data"]] == [0.94, 0.81]
    assert [result["content"][0]["text"] for result in response["data"]] == ["an astronaut adrift", "a robot dog"]
    assert [result["file_id"] for result in response["data"]] == ["abc123", "def456"]
    assert [result["filename"] for result in response["data"]] == ["abc123", "def456"]
    assert response["data"][0]["content"][0]["type"] == "text"


def test_response_reads_a_dotted_text_field_path():
    config, _, _ = _config(documents=[{"_id": 1, "metadata": {"body": "nested text"}, "score": 0.5}])

    response = _search(config, litellm_params={"mongodb_text_field": "metadata.body"})

    assert response["data"][0]["content"][0]["text"] == "nested text"


def test_response_tolerates_a_document_missing_the_text_field():
    config, _, _ = _config(documents=[{"_id": 1, "score": 0.5}])

    response = _search(config)

    assert response["data"][0]["content"][0]["text"] == ""


def test_response_tolerates_a_document_missing_a_score():
    config, _, _ = _config(documents=[{"_id": 1, "text": "no score"}])

    response = _search(config)

    assert response["data"][0]["score"] is None


def test_response_stringifies_a_non_string_document_id():
    config, _, _ = _config(documents=[{"_id": 12345, "text": "numeric id", "score": 0.5}])

    response = _search(config)

    assert response["data"][0]["file_id"] == "12345"


def test_search_requires_an_embedding_model():
    config, _, _ = _config()

    with pytest.raises(ValueError, match="litellm_embedding_model is required"):
        config.execute_search_vector_store_request(
            vector_store_id=INDEX,
            query="q",
            vector_store_search_optional_params={},
            litellm_logging_obj=MagicMock(),
            litellm_params={k: v for k, v in BASE_PARAMS.items() if k != "litellm_embedding_model"},
        )


def test_missing_embedding_model_message_names_the_field_being_searched():
    config, _, _ = _config()

    with pytest.raises(ValueError, match=r"embedded_movies\.embedding"):
        config.execute_search_vector_store_request(
            vector_store_id=INDEX,
            query="q",
            vector_store_search_optional_params={},
            litellm_logging_obj=MagicMock(),
            litellm_params={k: v for k, v in BASE_PARAMS.items() if k != "litellm_embedding_model"},
        )


def test_search_requires_a_connection_string():
    config, _, _ = _config()

    with pytest.raises(ValueError, match="mongodb_connection_string is required"):
        _search(config, litellm_params={"mongodb_connection_string": None})


@pytest.mark.parametrize("connection_string", ["postgres://host/db", "https://cluster.mongodb.net", "redis://host"])
def test_search_rejects_a_non_mongodb_connection_scheme(connection_string):
    config, _, _ = _config()

    with pytest.raises(ValueError, match="must start with 'mongodb://' or 'mongodb\\+srv://'"):
        _search(config, litellm_params={"mongodb_connection_string": connection_string})


def test_search_accepts_the_plain_mongodb_scheme():
    config, _, collection = _config()

    _search(config, litellm_params={"mongodb_connection_string": "mongodb://localhost:27017"})

    assert collection.pipeline is not None


def test_search_requires_a_database():
    config, _, _ = _config()

    with pytest.raises(ValueError, match="mongodb_database is required"):
        _search(config, litellm_params={"mongodb_database": None})


def test_search_requires_a_collection():
    config, _, _ = _config()

    with pytest.raises(ValueError, match="mongodb_collection is required"):
        _search(config, litellm_params={"mongodb_collection": None})


def test_search_rejects_filters_rather_than_silently_ignoring_them():
    config, _, _ = _config()

    with pytest.raises(ValueError, match="does not support the filters parameter"):
        _search(config, optional_params={"filters": {"genre": "sci-fi"}})


@pytest.mark.asyncio
async def test_async_search_rejects_filters_rather_than_silently_ignoring_them():
    config, _, _ = _async_config()

    with pytest.raises(ValueError, match="does not support the filters parameter"):
        await _asearch(config, optional_params={"filters": {"genre": "sci-fi"}})


@pytest.mark.parametrize("query", ["", "   ", "\n\t", []])
def test_search_rejects_an_empty_query(query):
    config, _, _ = _config()

    with pytest.raises(ValueError, match="query must not be empty"):
        _search(config, query=query)


def test_search_rejects_an_oversized_query():
    config, _, _ = _config()

    with pytest.raises(ValueError, match="at most 32000 characters"):
        _search(config, query="x" * 32_001)


def test_search_accepts_a_query_at_the_size_ceiling():
    config, _, collection = _config()

    _search(config, query="x" * 32_000)

    assert collection.pipeline is not None


@pytest.mark.parametrize("max_num_results", [0, -1, 51, 1000])
def test_search_rejects_out_of_range_max_num_results(max_num_results):
    config, _, _ = _config()

    with pytest.raises(ValueError, match="max_num_results must be between 1 and 50"):
        _search(config, optional_params={"max_num_results": max_num_results})


@pytest.mark.parametrize("max_num_results", [1, 50])
def test_search_allows_max_num_results_at_the_bounds(max_num_results):
    config, _, collection = _config()

    _search(config, optional_params={"max_num_results": max_num_results})

    assert _stage(collection, "$vectorSearch")["limit"] == max_num_results


def test_search_treats_an_explicit_null_max_num_results_as_the_default():
    config, _, collection = _config()

    _search(config, optional_params={"max_num_results": None})

    assert _stage(collection, "$vectorSearch")["limit"] == 10


def test_search_fails_when_the_embedding_model_returns_nothing():
    config, _, _ = _config(embedding=None)

    with pytest.raises(ValueError, match="returned no embedding"):
        _search(config)


def test_validation_runs_before_any_connection_is_opened():
    opened = []
    config = MongoDBVectorStoreConfig(
        embedding_fn=FakeEmbeddingFn([0.1]),
        sync_client_factory=lambda key: opened.append(key) or FakeClient(FakeCollection([])),
    )

    with pytest.raises(ValueError, match="query must not be empty"):
        _search(config, query="")

    assert opened == []


def test_create_vector_store_is_not_supported_and_says_why():
    config = MongoDBVectorStoreConfig()

    with pytest.raises(NotImplementedError, match="search-only"):
        config.transform_create_vector_store_request({}, "https://example.test")

    with pytest.raises(NotImplementedError, match="search-only"):
        config.transform_create_vector_store_response(httpx.Response(200))


def test_provider_config_manager_returns_the_mongodb_config():
    config = ProviderConfigManager.get_provider_vector_stores_config(LlmProviders.MONGODB)

    assert isinstance(config, MongoDBVectorStoreConfig)


@pytest.mark.asyncio
async def test_async_search_builds_the_same_pipeline_and_maps_the_response():
    documents = [{"_id": "abc123", "text": "an astronaut adrift", "score": 0.94}]
    config, client, collection = _async_config(documents=documents)

    response = await _asearch(config, optional_params={"max_num_results": 3})

    assert client.requested_database == "sample_mflix"
    assert client.database.requested_collection == "embedded_movies"
    assert _stage(collection, "$vectorSearch")["limit"] == 3
    assert _stage(collection, "$vectorSearch")["queryVector"] == [0.1, 0.2, 0.3]
    assert response["data"][0]["content"][0]["text"] == "an astronaut adrift"
    assert response["data"][0]["score"] == 0.94


@pytest.mark.asyncio
async def test_async_search_requires_an_embedding_model():
    config, _, _ = _async_config()

    with pytest.raises(ValueError, match="litellm_embedding_model is required"):
        await config.aexecute_search_vector_store_request(
            vector_store_id=INDEX,
            query="q",
            vector_store_search_optional_params={},
            litellm_logging_obj=MagicMock(),
            litellm_params={k: v for k, v in BASE_PARAMS.items() if k != "litellm_embedding_model"},
        )


class TestClientCache:
    def setup_method(self):
        reset_client_cache()

    def teardown_method(self):
        reset_client_cache()

    def _key(self, connection_string=CONNECTION_STRING, socket_timeout_ms=30_000):
        return MongoClientKey(
            connection_string=connection_string,
            connect_timeout_ms=10_000,
            socket_timeout_ms=socket_timeout_ms,
            server_selection_timeout_ms=10_000,
        )

    def test_the_same_connection_reuses_one_client(self):
        with patch("litellm.llms.mongodb.common_utils.import_sync_mongo_client") as importer:
            importer.return_value = lambda *args, **kwargs: MagicMock()

            first = get_sync_client(self._key())
            second = get_sync_client(self._key())

        assert first is second
        assert importer.return_value

    def test_a_different_connection_gets_its_own_client(self):
        with patch("litellm.llms.mongodb.common_utils.import_sync_mongo_client") as importer:
            importer.return_value = lambda *args, **kwargs: MagicMock()

            first = get_sync_client(self._key())
            second = get_sync_client(self._key(connection_string="mongodb://other.example.test"))

        assert first is not second

    def test_a_different_timeout_gets_its_own_client(self):
        with patch("litellm.llms.mongodb.common_utils.import_sync_mongo_client") as importer:
            importer.return_value = lambda *args, **kwargs: MagicMock()

            first = get_sync_client(self._key())
            second = get_sync_client(self._key(socket_timeout_ms=5_000))

        assert first is not second

    @pytest.mark.asyncio
    async def test_async_clients_are_cached_per_event_loop(self):
        with patch("litellm.llms.mongodb.common_utils.import_async_mongo_client") as importer:
            importer.return_value = lambda *args, **kwargs: MagicMock()

            first = get_async_client(self._key())
            second = get_async_client(self._key())

        assert first is second


class TestClientKeyDerivation:
    def test_no_timeout_uses_the_bounded_defaults(self):
        key = MongoDBVectorStoreConfig._client_key(_MongoDBSearchParams.model_validate(BASE_PARAMS), None)

        assert key.connect_timeout_ms == 10_000
        assert key.socket_timeout_ms == 30_000
        assert key.server_selection_timeout_ms == 10_000

    def test_a_numeric_timeout_bounds_the_connect_phase(self):
        key = MongoDBVectorStoreConfig._client_key(_MongoDBSearchParams.model_validate(BASE_PARAMS), 3.0)

        assert key.socket_timeout_ms == 3_000
        assert key.connect_timeout_ms == 3_000

    def test_an_httpx_timeout_maps_connect_and_read_separately(self):
        key = MongoDBVectorStoreConfig._client_key(
            _MongoDBSearchParams.model_validate(BASE_PARAMS), httpx.Timeout(connect=2.0, read=45.0, write=5.0, pool=5.0)
        )

        assert key.connect_timeout_ms == 2_000
        assert key.socket_timeout_ms == 45_000


class TestErrorTranslation:
    def _translate(self, error):
        return translate_mongo_error(error, index_name=INDEX, database="sample_mflix", collection="embedded_movies")

    def test_server_selection_timeout_points_at_the_atlas_access_list(self):
        from pymongo.errors import ServerSelectionTimeoutError

        translated = self._translate(ServerSelectionTimeoutError("no servers"))

        assert "IP access list" in str(translated)
        assert "paused cluster" in str(translated)

    def test_authentication_failure_points_at_the_connection_string_credentials(self):
        from pymongo.errors import OperationFailure

        translated = self._translate(OperationFailure("auth failed", code=18))

        assert "rejected the credentials" in str(translated)

    def test_unauthorized_points_at_the_database_user_permissions(self):
        from pymongo.errors import OperationFailure

        translated = self._translate(OperationFailure("not authorized", code=13))

        assert "sample_mflix.embedded_movies" in str(translated)

    def test_a_missing_index_names_the_index_and_the_collection(self):
        from pymongo.errors import OperationFailure

        translated = self._translate(OperationFailure("Index not found for name movies_vector_index", code=27))

        assert INDEX in str(translated)
        assert "READY" in str(translated)

    def test_a_dimension_mismatch_points_at_the_embedding_model(self):
        from pymongo.errors import OperationFailure

        translated = self._translate(OperationFailure("queryVector has 1536 dimensions, index expects 2048"))

        assert "litellm_embedding_model must be the same model" in str(translated)

    def test_an_unrecognised_operation_failure_still_names_the_target(self):
        from pymongo.errors import OperationFailure

        translated = self._translate(OperationFailure("something else entirely"))

        assert "sample_mflix.embedded_movies" in str(translated)
        assert INDEX in str(translated)

    def test_a_configuration_error_points_at_the_connection_string(self):
        from pymongo.errors import ConfigurationError

        translated = self._translate(ConfigurationError("bad uri"))

        assert "not a usable MongoDB connection string" in str(translated)

    def test_a_non_driver_error_is_returned_unchanged(self):
        original = RuntimeError("unrelated")

        assert self._translate(original) is original

    def test_search_surfaces_a_translated_driver_error(self):
        from pymongo.errors import ServerSelectionTimeoutError

        config, _, _ = _config(error=ServerSelectionTimeoutError("no servers"))

        with pytest.raises(ValueError, match="IP access list"):
            _search(config)

    @pytest.mark.asyncio
    async def test_async_search_surfaces_a_translated_driver_error(self):
        from pymongo.errors import OperationFailure

        config, _, _ = _async_config(error=OperationFailure("auth failed", code=18))

        with pytest.raises(ValueError, match="rejected the credentials"):
            await _asearch(config)


class TestMissingDriver:
    def test_the_sync_import_names_the_extra_to_install(self):
        from litellm.llms.mongodb.common_utils import import_sync_mongo_client

        with patch.dict(sys.modules, {"pymongo": None}):
            with pytest.raises(ValueError, match=r"pip install litellm\[mongodb\]"):
                import_sync_mongo_client()

    def test_the_async_import_names_the_extra_to_install(self):
        from litellm.llms.mongodb.common_utils import import_async_mongo_client

        with patch.dict(sys.modules, {"pymongo": None}):
            with pytest.raises(ValueError, match=r"pip install litellm\[mongodb\]"):
                import_async_mongo_client()

    def test_error_translation_degrades_gracefully_without_the_driver(self):
        original = RuntimeError("boom")

        with patch.dict(sys.modules, {"pymongo.errors": None}):
            assert translate_mongo_error(original, INDEX, "db", "col") is original
