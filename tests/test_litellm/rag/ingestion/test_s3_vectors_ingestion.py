from types import SimpleNamespace

import pytest

from litellm.rag.ingestion.s3_vectors_ingestion import S3VectorsRAGIngestion

STORE_ID_FORMAT_ERROR = "vector_store_id must be in format 'bucket_name:index_name'"
REQUEST_EMBEDDING_MODEL = "text-embedding-3-small"
STORE_EMBEDDING_MODEL = "text-embedding-3-large"
REQUEST_EMBEDDING = {"model": REQUEST_EMBEDDING_MODEL}


class _RecordingRouter:
    def __init__(self):
        self.embedding_models = []

    async def aembedding(self, model, input):
        self.embedding_models.append(model)
        return SimpleNamespace(data=[{"embedding": [0.1, 0.2]} for _ in input])


def _ingestion(embedding=REQUEST_EMBEDDING, router=None, **vector_store):
    vector_store_options = {"custom_llm_provider": "s3_vectors", "aws_region_name": "us-west-2", **vector_store}
    ingest_options = {"vector_store": vector_store_options} if embedding is None else {
        "embedding": embedding,
        "vector_store": vector_store_options,
    }
    return S3VectorsRAGIngestion(ingest_options=ingest_options, router=router)


@pytest.mark.asyncio
@pytest.mark.parametrize("store_model_key", ["embedding_model", "litellm_embedding_model"])
async def test_a_registered_store_embedding_model_wins_over_the_request_on_ingest(store_model_key):
    router = _RecordingRouter()
    ingestion = _ingestion(
        router=router, vector_store_id="my-embeddings:my-index", **{store_model_key: STORE_EMBEDDING_MODEL}
    )

    await ingestion.embed(["chunk one", "chunk two"])

    assert router.embedding_models == [STORE_EMBEDDING_MODEL]


@pytest.mark.asyncio
async def test_a_registered_store_embedding_model_is_used_when_the_request_names_none():
    router = _RecordingRouter()
    ingestion = _ingestion(
        embedding=None, router=router, vector_store_id="my-embeddings:my-index", embedding_model=STORE_EMBEDDING_MODEL
    )

    await ingestion.embed(["chunk"])

    assert router.embedding_models == [STORE_EMBEDDING_MODEL]


@pytest.mark.asyncio
@pytest.mark.parametrize("store_model", [{}, {"embedding_model": ""}])
async def test_the_request_embedding_model_is_kept_when_the_store_names_none(store_model):
    router = _RecordingRouter()
    ingestion = _ingestion(router=router, vector_store_id="my-embeddings:my-index", **store_model)

    await ingestion.embed(["chunk"])

    assert router.embedding_models == [REQUEST_EMBEDDING_MODEL]


def test_store_id_alone_names_the_bucket_and_index():
    ingestion = _ingestion(vector_store_id="my-embeddings:my-index")

    assert (ingestion.vector_bucket_name, ingestion.index_name) == ("my-embeddings", "my-index")


def test_store_id_without_a_colon_is_the_index_inside_the_given_bucket():
    ingestion = _ingestion(vector_store_id="my-index", vector_bucket_name="my-embeddings")

    assert (ingestion.vector_bucket_name, ingestion.index_name) == ("my-embeddings", "my-index")


def test_explicit_bucket_and_index_win_over_the_store_id():
    ingestion = _ingestion(vector_store_id="id-bucket:id-index", vector_bucket_name="my-bucket", index_name="docs")

    assert (ingestion.vector_bucket_name, ingestion.index_name) == ("my-bucket", "docs")


def test_bucket_alone_leaves_the_index_to_be_generated():
    ingestion = _ingestion(vector_bucket_name="my-embeddings")

    assert (ingestion.vector_bucket_name, ingestion.index_name) == ("my-embeddings", None)


@pytest.mark.parametrize(
    "vector_store",
    [{}, {"vector_store_id": "my-index"}, {"vector_store_id": "my-index", "vector_bucket_name": ""}],
)
def test_no_bucket_anywhere_is_rejected(vector_store):
    with pytest.raises(ValueError, match=STORE_ID_FORMAT_ERROR):
        _ingestion(**vector_store)


@pytest.mark.parametrize(
    "vector_store",
    [
        {"vector_store_id": "my-embeddings:"},
        {"vector_store_id": ":my-index"},
        {"vector_store_id": "my-embeddings:", "vector_bucket_name": "my-embeddings"},
    ],
)
def test_an_empty_bucket_or_index_in_the_store_id_is_rejected_instead_of_generating_an_index(vector_store):
    with pytest.raises(ValueError, match=STORE_ID_FORMAT_ERROR):
        _ingestion(**vector_store)
