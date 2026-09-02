import asyncio
import gc
import sys
import weakref
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest

from litellm.exceptions import BadRequestError, Timeout
from litellm.llms.mongodb.common_utils import (
    MongoClientKey,
    index_not_ready_error,
    missing_index_error,
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


READY_INDEX = [{"name": INDEX, "status": "READY", "queryable": True}]


class RecordingClient:
    """Stands in for pymongo's client class so the cache tests inject a fake rather than
    patching the importer, and so they can assert what the client was actually built with."""

    def __init__(self, connection_string, **kwargs):
        self.connection_string = connection_string
        self.kwargs = kwargs


class FakeCollection:
    def __init__(self, documents, error=None, search_indexes=None):
        self.documents = documents
        self.error = error
        self.search_indexes = READY_INDEX if search_indexes is None else search_indexes
        self.pipeline = None
        self.listed_indexes = []

    def aggregate(self, pipeline):
        self.pipeline = pipeline
        if self.error is not None:
            raise self.error
        return iter(self.documents)

    def list_search_indexes(self, name):
        self.listed_indexes.append(name)
        return iter(self.search_indexes)


class FakeAsyncCollection(FakeCollection):
    async def aggregate(self, pipeline):
        self.pipeline = pipeline
        if self.error is not None:
            raise self.error

        async def cursor():
            for document in self.documents:
                yield document

        return cursor()

    async def list_search_indexes(self, name):
        self.listed_indexes.append(name)

        async def cursor():
            for entry in self.search_indexes:
                yield entry

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


def _config(documents=(), embedding=(0.1, 0.2, 0.3), error=None, search_indexes=None):
    collection = FakeCollection(list(documents), error, search_indexes)
    client = FakeClient(collection)
    config = MongoDBVectorStoreConfig(
        embedding_fn=FakeEmbeddingFn(list(embedding) if embedding is not None else None),
        sync_client_factory=lambda key: client,
    )
    return config, client, collection


def _async_config(documents=(), embedding=(0.1, 0.2, 0.3), error=None, search_indexes=None):
    collection = FakeAsyncCollection(list(documents), error, search_indexes)
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
        "queryVector": (0.1, 0.2, 0.3),
        "numCandidates": 100,
        "limit": 5,
    }


def test_the_pipeline_reaches_pymongo_as_a_list():
    """pymongo's common.validate_list rejects any other sequence with
    'pipeline must be a list, not <class ...>', so the outer container is part of the contract."""
    config, _, collection = _config()

    _search(config)

    assert isinstance(collection.pipeline, list)


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

    with pytest.raises(BadRequestError, match="mongodb_num_candidates"):
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


def test_a_dotted_path_resolves_three_levels_deep():
    config, _, _ = _config(documents=[{"_id": 1, "a": {"b": {"c": "deep text"}}, "score": 0.5}])

    response = _search(config, litellm_params={"mongodb_text_field": "a.b.c"})

    assert response["data"][0]["content"][0]["text"] == "deep text"


def test_a_dotted_path_that_runs_through_a_scalar_counts_as_absent():
    """Walking 'plot.nope' when plot is a string must report the misconfiguration, not
    stringify the scalar and hand the model text from the wrong field."""
    config, _, _ = _config(documents=[{"_id": 1, "plot": "a plain string", "score": 0.5}])

    with pytest.raises(BadRequestError, match=r"has a 'plot\.nope' field"):
        _search(config, litellm_params={"mongodb_text_field": "plot.nope"})


def test_a_non_string_text_field_is_stringified():
    config, _, _ = _config(documents=[{"_id": 1, "year": 1979, "score": 0.5}])

    response = _search(config, litellm_params={"mongodb_text_field": "year"})

    assert response["data"][0]["content"][0]["text"] == "1979"


def test_a_null_text_field_counts_as_absent():
    config, _, _ = _config(documents=[{"_id": 1, "text": None, "score": 0.5}])

    with pytest.raises(BadRequestError, match="has a 'text' field"):
        _search(config)


def test_response_tolerates_a_sparse_document_missing_the_text_field():
    config, _, _ = _config(documents=[{"_id": 1, "score": 0.5}, {"_id": 2, "text": "has text", "score": 0.4}])

    response = _search(config)

    assert response["data"][0]["content"][0]["text"] == ""
    assert response["data"][1]["content"][0]["text"] == "has text"


def test_a_present_but_empty_text_field_is_not_treated_as_a_misconfiguration():
    config, _, _ = _config(documents=[{"_id": 1, "text": "", "score": 0.5}])

    response = _search(config)

    assert response["data"][0]["content"][0]["text"] == ""


def test_matches_that_all_lack_the_text_field_name_the_setting_to_fix():
    """Atlas matches on the vector, so a mistyped mongodb_text_field returns confidently
    scored results whose content is empty and hands the model an empty context."""
    config, _, _ = _config(documents=[{"_id": 1, "score": 0.9}, {"_id": 2, "score": 0.8}])

    with pytest.raises(BadRequestError, match="mongodb_text_field"):
        _search(config)


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

    with pytest.raises(BadRequestError, match="litellm_embedding_model is required"):
        config.execute_search_vector_store_request(
            vector_store_id=INDEX,
            query="q",
            vector_store_search_optional_params={},
            litellm_logging_obj=MagicMock(),
            litellm_params={k: v for k, v in BASE_PARAMS.items() if k != "litellm_embedding_model"},
        )


def test_missing_embedding_model_message_names_the_field_being_searched():
    config, _, _ = _config()

    with pytest.raises(BadRequestError, match=r"embedded_movies\.embedding"):
        config.execute_search_vector_store_request(
            vector_store_id=INDEX,
            query="q",
            vector_store_search_optional_params={},
            litellm_logging_obj=MagicMock(),
            litellm_params={k: v for k, v in BASE_PARAMS.items() if k != "litellm_embedding_model"},
        )


def test_search_requires_a_connection_string():
    config, _, _ = _config()

    with pytest.raises(BadRequestError, match="mongodb_connection_string is required"):
        _search(config, litellm_params={"mongodb_connection_string": None})


@pytest.mark.parametrize("connection_string", ["postgres://host/db", "https://cluster.mongodb.net", "redis://host"])
def test_search_rejects_a_non_mongodb_connection_scheme(connection_string):
    config, _, _ = _config()

    with pytest.raises(BadRequestError, match="must start with 'mongodb://' or 'mongodb\\+srv://'"):
        _search(config, litellm_params={"mongodb_connection_string": connection_string})


def test_search_accepts_the_plain_mongodb_scheme():
    config, _, collection = _config()

    _search(config, litellm_params={"mongodb_connection_string": "mongodb://localhost:27017"})

    assert collection.pipeline is not None


def test_search_requires_a_database():
    config, _, _ = _config()

    with pytest.raises(BadRequestError, match="mongodb_database is required"):
        _search(config, litellm_params={"mongodb_database": None})


def test_search_requires_a_collection():
    config, _, _ = _config()

    with pytest.raises(BadRequestError, match="mongodb_collection is required"):
        _search(config, litellm_params={"mongodb_collection": None})


def test_search_rejects_filters_rather_than_silently_ignoring_them():
    config, _, _ = _config()

    with pytest.raises(BadRequestError, match="does not support the filters parameter"):
        _search(config, optional_params={"filters": {"genre": "sci-fi"}})


@pytest.mark.asyncio
async def test_async_search_rejects_filters_rather_than_silently_ignoring_them():
    config, _, _ = _async_config()

    with pytest.raises(BadRequestError, match="does not support the filters parameter"):
        await _asearch(config, optional_params={"filters": {"genre": "sci-fi"}})


@pytest.mark.parametrize("query", ["", "   ", "\n\t", []])
def test_search_rejects_an_empty_query(query):
    config, _, _ = _config()

    with pytest.raises(BadRequestError, match="query must not be empty"):
        _search(config, query=query)


def test_search_rejects_an_oversized_query():
    config, _, _ = _config()

    with pytest.raises(BadRequestError, match="at most 32000 characters"):
        _search(config, query="x" * 32_001)


def test_search_accepts_a_query_at_the_size_ceiling():
    config, _, collection = _config()

    _search(config, query="x" * 32_000)

    assert collection.pipeline is not None


@pytest.mark.parametrize("max_num_results", [0, -1, 51, 1000])
def test_search_rejects_out_of_range_max_num_results(max_num_results):
    config, _, _ = _config()

    with pytest.raises(BadRequestError, match="max_num_results must be between 1 and 50"):
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

    with pytest.raises(BadRequestError, match="returned no embedding"):
        _search(config)


def test_validation_runs_before_any_connection_is_opened():
    opened = []
    config = MongoDBVectorStoreConfig(
        embedding_fn=FakeEmbeddingFn([0.1]),
        sync_client_factory=lambda key: opened.append(key) or FakeClient(FakeCollection([])),
    )

    with pytest.raises(BadRequestError, match="query must not be empty"):
        _search(config, query="")

    assert opened == []


def test_create_vector_store_is_not_supported_and_says_why():
    """litellm.exception_type only passes its own exception types through untouched, so a
    NotImplementedError here reaches the caller as APIConnectionError, which the proxy serves
    as a 500 with a traceback. Refusing an unsupported operation is a client error."""
    config = MongoDBVectorStoreConfig()

    with pytest.raises(BadRequestError, match="search-only"):
        config.transform_create_vector_store_request({}, "https://example.test")

    with pytest.raises(BadRequestError, match="search-only"):
        config.transform_create_vector_store_response(httpx.Response(200))


def test_the_create_refusal_survives_the_public_sdk_error_wrapper():
    import litellm

    with pytest.raises(BadRequestError) as raised:
        litellm.vector_stores.create(custom_llm_provider="mongodb", name="anything")

    assert "search-only" in str(raised.value)


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
    assert _stage(collection, "$vectorSearch")["queryVector"] == (0.1, 0.2, 0.3)
    assert response["data"][0]["content"][0]["text"] == "an astronaut adrift"
    assert response["data"][0]["score"] == 0.94


@pytest.mark.asyncio
async def test_async_search_requires_an_embedding_model():
    config, _, _ = _async_config()

    with pytest.raises(BadRequestError, match="litellm_embedding_model is required"):
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
        first = get_sync_client(self._key(), RecordingClient)
        second = get_sync_client(self._key(), RecordingClient)

        assert first is second
        assert first.connection_string == CONNECTION_STRING
        assert first.kwargs["socketTimeoutMS"] == 30_000
        assert first.kwargs["connectTimeoutMS"] == 10_000
        assert first.kwargs["appname"] == "litellm"

    def test_a_different_connection_gets_its_own_client(self):
        first = get_sync_client(self._key(), RecordingClient)
        second = get_sync_client(self._key(connection_string="mongodb://other.example.test"), RecordingClient)

        assert first is not second
        assert second.connection_string == "mongodb://other.example.test"

    def test_a_different_timeout_gets_its_own_client(self):
        first = get_sync_client(self._key(), RecordingClient)
        second = get_sync_client(self._key(socket_timeout_ms=5_000), RecordingClient)

        assert first is not second
        assert second.kwargs["socketTimeoutMS"] == 5_000

    @pytest.mark.asyncio
    async def test_async_clients_are_cached_per_event_loop(self):
        first = get_async_client(self._key(), RecordingClient)
        second = get_async_client(self._key(), RecordingClient)

        assert first is second
        assert first.connection_string == CONNECTION_STRING


    def test_a_new_loop_never_inherits_a_closed_loop_client(self):
        """CPython recycles id() so aggressively that a fresh event loop almost always lands on
        the id of one already collected: measured at 37 of 40 rounds. Keying the cache on the id
        alone therefore hands the new loop an AsyncMongoClient bound to a closed loop, and every
        operation on it raises "Event loop is closed"."""

        class LoopAgnosticClient:
            """Holds no reference to the loop, unlike pymongo's, whose own reference happens to
            keep ids from being recycled and hides the bug until the cache fills."""

            def __init__(self, *args, **kwargs):
                self.built_on = None

        key = self._key()
        clients_handed_out = []

        async def fetch():
            return get_async_client(key, LoopAgnosticClient)

        for _ in range(20):
            loop = asyncio.new_event_loop()
            client = loop.run_until_complete(fetch())
            clients_handed_out.append((client, client.built_on, loop.is_closed()))
            client.built_on = weakref.ref(loop)
            loop.close()
            del loop
            gc.collect()

        stale = [
            handed_out
            for client, built_on, _ in clients_handed_out
            if built_on is not None and (built_on() is None or built_on().is_closed())
            for handed_out in (client,)
        ]
        assert stale == [], f"{len(stale)} of 20 loops were handed a client built on a closed loop"


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

    def test_a_short_timeout_also_shortens_server_selection(self):
        """Server selection runs before the connect attempt, so leaving it at the 10s default
        would let a caller asking for a 3s budget block for 10s before anything is tried."""
        key = MongoDBVectorStoreConfig._client_key(_MongoDBSearchParams.model_validate(BASE_PARAMS), 3.0)

        assert key.server_selection_timeout_ms == 3_000

    def test_a_generous_timeout_does_not_raise_server_selection_above_the_default(self):
        key = MongoDBVectorStoreConfig._client_key(_MongoDBSearchParams.model_validate(BASE_PARAMS), 120.0)

        assert key.socket_timeout_ms == 120_000
        assert key.server_selection_timeout_ms == 10_000

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

    def test_code_13_alone_is_enough_without_a_recognisable_message(self):
        """The other unauthorized case carries "not authorized", which the message markers also
        match, so it cannot tell whether the code is still being checked at all."""
        from pymongo.errors import OperationFailure

        translated = self._translate(OperationFailure("user lacks privileges on this namespace", code=13))

        assert "rejected the credentials" in str(translated)
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

        with pytest.raises(Timeout, match="IP access list"):
            _search(config)

    @pytest.mark.asyncio
    async def test_async_search_surfaces_a_translated_driver_error(self):
        from pymongo.errors import OperationFailure

        config, _, _ = _async_config(error=OperationFailure("auth failed", code=18))

        with pytest.raises(BadRequestError, match="rejected the credentials"):
            await _asearch(config)


class TestMissingDriver:
    def test_the_sync_import_names_the_extra_to_install(self):
        from litellm.llms.mongodb.common_utils import import_sync_mongo_client

        with patch.dict(sys.modules, {"pymongo": None}):
            with pytest.raises(BadRequestError, match=r"pip install litellm\[mongodb\]"):
                import_sync_mongo_client()

    def test_the_async_import_names_the_extra_to_install(self):
        from litellm.llms.mongodb.common_utils import import_async_mongo_client

        with patch.dict(sys.modules, {"pymongo": None}):
            with pytest.raises(BadRequestError, match=r"pip install litellm\[mongodb\]"):
                import_async_mongo_client()

    def test_error_translation_degrades_gracefully_without_the_driver(self):
        original = RuntimeError("boom")

        with patch.dict(sys.modules, {"pymongo.errors": None}):
            assert translate_mongo_error(original, INDEX, "db", "col") is original


class TestEmptyResultsAreDisambiguated:
    """$vectorSearch returns zero documents for a missing database, collection or index just as it
    does for a query that matched nothing, so an empty result set is checked against the index
    catalogue before it is reported as 'no matches'."""

    def test_a_missing_index_becomes_an_error_rather_than_an_empty_page(self):
        config, _, collection = _config(documents=[], search_indexes=[])

        with pytest.raises(BadRequestError, match="No queryable Atlas Vector Search index"):
            _search(config)

        assert collection.listed_indexes == [INDEX]

    def test_the_missing_index_error_explains_why_mongodb_reported_no_results(self):
        config, _, _ = _config(documents=[], search_indexes=[])

        with pytest.raises(BadRequestError, match="returns no results rather than an error"):
            _search(config)

    def test_an_index_still_building_becomes_an_error_naming_its_status(self):
        config, _, _ = _config(
            documents=[], search_indexes=[{"name": INDEX, "status": "PENDING", "queryable": False}]
        )

        with pytest.raises(BadRequestError, match="not queryable yet; its status is PENDING"):
            _search(config)

    def test_a_genuine_no_match_against_a_ready_index_returns_an_empty_page(self):
        config, _, collection = _config(documents=[])

        response = _search(config)

        assert response["data"] == []
        assert response["object"] == "vector_store.search_results.page"
        assert collection.listed_indexes == [INDEX]

    def test_the_catalogue_is_not_consulted_when_the_search_returned_hits(self):
        config, _, collection = _config(documents=[{"_id": 1, "text": "hit", "score": 0.9}])

        _search(config)

        assert collection.listed_indexes == []

    @pytest.mark.asyncio
    async def test_async_missing_index_becomes_an_error_rather_than_an_empty_page(self):
        config, _, collection = _async_config(documents=[], search_indexes=[])

        with pytest.raises(BadRequestError, match="No queryable Atlas Vector Search index"):
            await _asearch(config)

        assert collection.listed_indexes == [INDEX]

    @pytest.mark.asyncio
    async def test_async_index_still_building_becomes_an_error_naming_its_status(self):
        config, _, _ = _async_config(
            documents=[], search_indexes=[{"name": INDEX, "status": "PENDING", "queryable": False}]
        )

        with pytest.raises(BadRequestError, match="not queryable yet; its status is PENDING"):
            await _asearch(config)

    @pytest.mark.asyncio
    async def test_async_genuine_no_match_returns_an_empty_page(self):
        config, _, _ = _async_config(documents=[])

        response = await _asearch(config)

        assert response["data"] == []

    @pytest.mark.asyncio
    async def test_async_catalogue_is_not_consulted_when_the_search_returned_hits(self):
        config, _, collection = _async_config(documents=[{"_id": 1, "text": "hit", "score": 0.9}])

        await _asearch(config)

        assert collection.listed_indexes == []

    def test_a_failure_while_checking_the_catalogue_is_translated_too(self):
        from pymongo.errors import OperationFailure

        class ExplodingCollection(FakeCollection):
            def list_search_indexes(self, name):
                raise OperationFailure("not authorized", code=13)

        collection = ExplodingCollection([], None, [])
        config = MongoDBVectorStoreConfig(
            embedding_fn=FakeEmbeddingFn([0.1]),
            sync_client_factory=lambda key: FakeClient(collection),
        )

        with pytest.raises(BadRequestError, match="lacks read access"):
            _search(config)


class TestAtlasPlanExecutorErrors:
    """Atlas reports a wrong vector path and a dimension mismatch through the same error code, so
    each one has to be told apart by its message or both come back as a generic index failure."""

    def _translate(self, message):
        from pymongo.errors import OperationFailure

        return translate_mongo_error(
            OperationFailure(message, code=8),
            index_name=INDEX,
            database="sample_mflix",
            collection="embedded_movies",
        )

    def test_a_wrong_vector_path_points_at_the_embedding_field_setting(self):
        translated = self._translate(
            "PlanExecutor error during aggregation :: caused by :: nope is not indexed as vector"
        )

        assert "mongodb_embedding_field names a field" in str(translated)

    def test_a_dimension_mismatch_is_not_reported_as_a_wrong_path(self):
        translated = self._translate(
            "PlanExecutor error during aggregation :: caused by :: vector field is indexed with "
            "1536 dimensions but queried with 3072"
        )

        assert "does not match the vector dimensions" in str(translated)
        assert "mongodb_embedding_field" not in str(translated)


class TestErrorsCarryTheRightHttpStatus:
    """litellm.exception_type passes a litellm exception through untouched but wraps anything
    else into APIConnectionError, which the proxy serves as a 500 with a Python traceback in the
    body. A misconfigured connection string is the caller's to fix, so it has to arrive as a 400.
    """

    @pytest.mark.parametrize(
        "invoke",
        [
            pytest.param(lambda: _search(_config()[0], query="  "), id="empty-query"),
            pytest.param(
                lambda: _search(_config()[0], optional_params={"max_num_results": 999}),
                id="max-num-results-out-of-range",
            ),
            pytest.param(
                lambda: _search(_config()[0], optional_params={"filters": {"genre": "Action"}}),
                id="unsupported-filters",
            ),
            pytest.param(
                lambda: _search(_config()[0], litellm_params={"mongodb_connection_string": "postgres://host/db"}),
                id="wrong-uri-scheme",
            ),
            pytest.param(
                lambda: _search(_config()[0], litellm_params={"mongodb_database": None}), id="missing-database"
            ),
            pytest.param(
                lambda: _search(_config()[0], litellm_params={"litellm_embedding_model": None}),
                id="missing-embedding-model",
            ),
        ],
    )
    def test_configuration_failures_are_400(self, invoke):
        with pytest.raises(BadRequestError) as excinfo:
            invoke()
        assert excinfo.value.status_code == 400
        assert excinfo.value.llm_provider == "mongodb"

    def test_missing_index_is_400(self):
        error = missing_index_error("idx", "db", "coll")
        assert error.status_code == 400
        assert error.llm_provider == "mongodb"

    def test_index_still_building_is_400(self):
        error = index_not_ready_error("idx", "db", "coll", "PENDING")
        assert error.status_code == 400

    def test_unreachable_deployment_is_a_timeout_not_a_bad_request(self):
        from pymongo.errors import ServerSelectionTimeoutError

        translated = translate_mongo_error(
            ServerSelectionTimeoutError("no servers"), index_name="idx", database="db", collection="coll"
        )
        assert isinstance(translated, Timeout)
        assert translated.status_code == 408

    def test_query_execution_timeout_is_a_timeout(self):
        from pymongo.errors import ExecutionTimeout

        translated = translate_mongo_error(
            ExecutionTimeout("too slow"), index_name="idx", database="db", collection="coll"
        )
        assert isinstance(translated, Timeout)
        assert translated.status_code == 408

    def test_unrecognised_errors_are_not_relabelled_as_bad_requests(self):
        original = RuntimeError("something else entirely")
        assert (
            translate_mongo_error(original, index_name="idx", database="db", collection="coll")
            is original
        )


def test_atlas_rejected_credentials_are_named_even_though_the_code_is_8000():
    """Atlas answers a wrong password with code 8000 "AtlasError", not the 18 that a
    self-hosted deployment returns, so a code-only check reports it as a generic
    rejected search and never tells the caller to look at their connection string."""
    from pymongo.errors import OperationFailure

    error = OperationFailure(
        "bad auth : authentication failed",
        code=8000,
        details={"ok": 0, "errmsg": "bad auth : authentication failed", "code": 8000, "codeName": "AtlasError"},
    )
    translated = translate_mongo_error(error, index_name="idx", database="sample_mflix", collection="embedded_movies")

    assert isinstance(translated, BadRequestError)
    assert "mongodb_connection_string" in str(translated)
    assert "sample_mflix.embedded_movies" in str(translated)


def test_a_rejected_search_that_is_not_an_auth_failure_keeps_the_generic_message():
    from pymongo.errors import OperationFailure

    error = OperationFailure("PlanExecutor error", code=8, details={"errmsg": "PlanExecutor error"})
    translated = translate_mongo_error(error, index_name="idx", database="db", collection="coll")

    assert "mongodb_connection_string" not in str(translated)


class TestUnrecognisedParameters:
    """litellm_params carries plenty of keys this provider does not own, so the params model has
    to ignore extras. That turns a mistyped mongodb_collection into 'mongodb_collection is
    required', pointing the reader at a key they can see they have set."""

    def test_a_mistyped_parameter_is_named(self):
        config, _, _ = _config()

        with pytest.raises(BadRequestError, match="mongodb_collectoin"):
            _search(config, litellm_params={"mongodb_collectoin": "embedded_movies"})

    def test_the_supported_names_are_listed(self):
        config, _, _ = _config()

        with pytest.raises(BadRequestError, match="mongodb_connection_string"):
            _search(config, litellm_params={"mongodb_databse": "sample_mflix"})

    def test_unrelated_litellm_params_are_still_ignored(self):
        config, _, _ = _config(documents=[{"_id": 1, "text": "hit", "score": 0.9}])

        response = _search(
            config,
            litellm_params={"use_litellm_proxy": False, "use_in_pass_through": False, "vector_store_id": "x"},
        )

        assert len(response["data"]) == 1

    @pytest.mark.asyncio
    async def test_the_async_path_rejects_them_too(self):
        config, _, _ = _async_config()

        with pytest.raises(BadRequestError, match="mongodb_collectoin"):
            await _asearch(config, litellm_params={"mongodb_collectoin": "embedded_movies"})


class TestClientConstructionFailures:
    """Building the client parses the URI and, for mongodb+srv://, performs a DNS SRV lookup, so it
    fails on exactly the inputs a user is most likely to get wrong. Constructing it outside the
    translation boundary let those escape as raw pymongo errors, which litellm.exception_type then
    wrapped into a 500 with a traceback in the body."""

    def _config_that_fails_to_connect(self, error):
        def factory(_key):
            raise error

        return MongoDBVectorStoreConfig(
            embedding_fn=FakeEmbeddingFn([0.1, 0.2, 0.3]), sync_client_factory=factory
        )

    def _async_config_that_fails_to_connect(self, error):
        def factory(_key):
            raise error

        return MongoDBVectorStoreConfig(
            aembedding_fn=FakeAsyncEmbeddingFn([0.1, 0.2, 0.3]), async_client_factory=factory
        )

    def test_a_malformed_uri_is_a_bad_request_not_a_500(self):
        from pymongo.errors import InvalidURI

        config = self._config_that_fails_to_connect(InvalidURI("Invalid URI scheme"))

        with pytest.raises(BadRequestError, match="not a usable MongoDB connection string"):
            _search(config)

    def test_an_unresolvable_cluster_name_says_so(self):
        from pymongo.errors import ConfigurationError

        config = self._config_that_fails_to_connect(ConfigurationError("The DNS query name does not exist"))

        with pytest.raises(BadRequestError, match="does not exist in DNS"):
            _search(config)

    def test_a_dns_lookup_that_ran_out_of_time_is_a_timeout(self):
        from pymongo.errors import ConfigurationError

        config = self._config_that_fails_to_connect(
            ConfigurationError("The resolution lifetime expired after 0.291 seconds")
        )

        with pytest.raises(Timeout, match="did not finish in time"):
            _search(config)

    @pytest.mark.asyncio
    async def test_the_async_path_translates_them_too(self):
        from pymongo.errors import InvalidURI

        config = self._async_config_that_fails_to_connect(InvalidURI("Invalid URI scheme"))

        with pytest.raises(BadRequestError, match="not a usable MongoDB connection string"):
            await _asearch(config)
