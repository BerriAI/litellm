"""Fixtures for the Milvus vector-store e2e suite.

The shared lifecycle (resources / scoped_key), proxy-liveness skip, and the e2e
marker come from the parent tests/e2e/conftest.py. This suite adds two things:

  - a `client` fixture that skips the whole suite unless the Milvus + OpenAI
    credentials it needs are present (skip on environment, never silently pass)
  - a session-scoped `seeded_store` fixture that stands up a real, populated
    managed vector store once, hands it to the tests, and tears it down after.
"""

from __future__ import annotations

import time
from typing import Iterator

import pytest

from e2e_config import unique_marker
from e2e_http import is_ok, unwrap
from milvus_client import (
    MilvusEntity,
    SeededStore,
    VectorStoreClient,
    build_client,
    build_corpus,
    credentials_reason,
)

SEARCHABLE_TIMEOUT_SECONDS = 90.0
SEARCHABLE_POLL_SECONDS = 3.0
PROBE_QUERY = "coral reef"


@pytest.fixture(scope="session")
def client() -> VectorStoreClient:
    reason = credentials_reason()
    if reason is not None:
        pytest.skip(reason)
    return build_client()


@pytest.fixture(scope="session")
def seeded_store(client: VectorStoreClient) -> Iterator[SeededStore]:
    marker = unique_marker()
    vector_store_id = f"e2e_milvus_{marker}"
    docs, secret_code, code_doc_id = build_corpus(marker)

    client.create_collection(vector_store_id)
    try:
        ids = sorted(docs)
        vectors = client.embed([docs[i].text for i in ids])
        entities = [
            MilvusEntity(id=i, vector=vector, text=docs[i].text, category=docs[i].category)
            for i, vector in zip(ids, vectors)
        ]
        client.insert(vector_store_id, entities)

        registration = client.register_store(
            vector_store_id, client.store_litellm_params()
        )
        if not is_ok(registration):
            pytest.skip(
                "managed vector store registration failed (needs a proxy with a "
                f"database and vector-store feature access): {registration}"
            )

        _wait_until_searchable(client, vector_store_id)

        yield SeededStore(
            vector_store_id=vector_store_id,
            docs=docs,
            secret_code=secret_code,
            # Marker in the prompt so the question text varies per run, defeating
            # any prompt-level response cache (OpenAI's or the proxy's) that would
            # otherwise return a previous run's secret code.
            code_query=(
                f"[run {marker}] What is the Project Nimbus access code? "
                "Reply with only the code."
            ),
            code_doc_id=code_doc_id,
        )
        client.delete_store(vector_store_id)
    finally:
        client.drop_collection(vector_store_id)


def _wait_until_searchable(client: VectorStoreClient, vector_store_id: str) -> None:
    """Newly inserted Milvus entities are not queryable until indexed/loaded, so
    poll a probe search until it returns data rather than sleeping a fixed guess.
    A store that never becomes searchable is an environment problem, not a test
    failure, so time out into a skip."""
    deadline = time.monotonic() + SEARCHABLE_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        result = client.search(vector_store_id, PROBE_QUERY)
        if is_ok(result) and unwrap(result).data:
            return
        time.sleep(SEARCHABLE_POLL_SECONDS)
    pytest.skip(
        f"seeded documents never became searchable in {vector_store_id} "
        f"within {SEARCHABLE_TIMEOUT_SECONDS:.0f}s"
    )
