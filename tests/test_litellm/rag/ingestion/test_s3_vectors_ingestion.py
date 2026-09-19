import pytest

from litellm.rag.ingestion.s3_vectors_ingestion import S3VectorsRAGIngestion

STORE_ID_FORMAT_ERROR = "vector_store_id must be in format 'bucket_name:index_name'"


def _ingestion(**vector_store):
    return S3VectorsRAGIngestion(
        ingest_options={
            "embedding": {"model": "text-embedding-3-small"},
            "vector_store": {"custom_llm_provider": "s3_vectors", "aws_region_name": "us-west-2", **vector_store},
        }
    )


def test_store_id_alone_names_the_bucket_and_index():
    """
    Regression for LIT-7956: a registered S3 Vectors store carries only its
    "bucket:index" id, and the proxy no longer forwards the caller's bucket and
    index for a managed store, so the ingestion must read both from the id.
    """
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
