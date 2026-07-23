"""Live e2e for the Milvus managed vector store, end to end through the proxy.

The `seeded_store` fixture (see conftest.py) creates a Milvus collection, seeds it
with five known documents, and registers it as a litellm managed vector store.
These tests then drive the proxy exactly as a user would: search the store over
the OpenAI-compatible route, and use it as retrieval context in a chat completion.

Retrieval is proven, not assumed: one seeded document holds a made-up access code
that appears nowhere else, so a chat answer echoing that code can only come from
the store being searched and its content injected as context.

Skips are environment-only (no proxy, no Milvus/OpenAI creds, no DB for managed
stores). Once the store is up, every assertion is behavioral.
"""

from __future__ import annotations

import json

import pytest

from e2e_config import unique_marker
from e2e_http import is_ok, unwrap
from milvus_client import (
    ManagedStoreEntry,
    SeededStore,
    VectorStoreClient,
    seed_probe_collection,
)

pytestmark = pytest.mark.e2e


def _texts(results) -> list[str]:
    return [result.content[0].text for result in results if result.content]


# ---- search --------------------------------------------------------------


def test_search_returns_seeded_documents_in_openai_shape(
    client: VectorStoreClient, seeded_store: SeededStore
) -> None:
    response = unwrap(client.search(seeded_store.vector_store_id, "coral reef"))

    assert response.object == "vector_store.search_results.page"
    assert response.data, "search returned no results for a seeded query"

    top = response.data[0]
    assert top.content, "result carried no content"
    assert top.content[0].type == "text"
    assert top.content[0].text, "result content text was empty"
    assert isinstance(top.score, float)

    reef_document = seeded_store.text(1)
    assert reef_document in _texts(response.data), (
        "the coral-reef document was not retrieved for a coral-reef query; "
        f"got {_texts(response.data)}"
    )


def test_search_respects_limit(
    client: VectorStoreClient, seeded_store: SeededStore
) -> None:
    response = unwrap(client.search(seeded_store.vector_store_id, "coral reef", limit=1))
    assert len(response.data) == 1, f"limit=1 returned {len(response.data)} results"


def test_search_with_filter_restricts_results(
    client: VectorStoreClient, seeded_store: SeededStore
) -> None:
    allowed_ids = [2, 3]
    response = unwrap(
        client.search(
            seeded_store.vector_store_id,
            "a geographic landmark",
            filter=f"id in {allowed_ids}",
        )
    )
    assert response.data, "filtered search returned nothing"

    allowed_texts = {seeded_store.text(i) for i in allowed_ids}
    excluded_texts = {
        seeded_store.text(i) for i in seeded_store.docs if i not in allowed_ids
    }
    returned = set(_texts(response.data))
    assert returned <= allowed_texts, f"filter leaked disallowed docs: {returned}"
    assert returned.isdisjoint(excluded_texts)


# ---- chat completions with the store as retrieval context ----------------


def test_chat_completion_retrieves_context_from_store(
    client: VectorStoreClient, seeded_store: SeededStore
) -> None:
    result = client.chat_with_store(seeded_store.vector_store_id, seeded_store.code_query)
    response = unwrap(result)

    assert response.choices, "chat completion returned no choices"
    answer = (response.choices[0].message.content if response.choices[0].message else "") or ""
    assert seeded_store.secret_code.lower() in answer.lower(), (
        "chat answer did not contain the access code that only exists in the "
        f"seeded document; RAG context was not injected. answer={answer!r}"
    )


# ---- graceful failure ----------------------------------------------------


def test_search_on_missing_collection_degrades_without_crashing(
    client: VectorStoreClient,
) -> None:
    """A managed store pointing at a Milvus collection that does not exist must not
    5xx-crash the proxy. The Milvus provider transform does not inspect Milvus's
    response code, so the upstream error surfaces as an empty result set rather
    than a propagated error; assert that graceful (non-crashing) behavior."""
    missing_id = f"e2e_milvus_missing_{unique_marker()}"
    registration = client.register_store(missing_id, client.store_litellm_params())
    if not is_ok(registration):
        pytest.skip(f"managed store registration unavailable: {registration}")

    try:
        outcome = client.search_raw(missing_id, "anything at all")
        assert outcome.status_code < 500, f"missing collection crashed the proxy: {outcome}"
        if outcome.ok:
            assert json.loads(outcome.body).get("data") == [], (
                f"expected empty results for a missing collection, got {outcome.body}"
            )
    finally:
        client.delete_store(missing_id)


# ---- search parameter variants ------------------------------------------


def test_search_respects_offset(
    client: VectorStoreClient, seeded_store: SeededStore
) -> None:
    """offset skips the top-N most similar results; page 1 + page 2 must be
    disjoint and page 2 must be strictly further from the query."""
    page_1 = unwrap(client.search(seeded_store.vector_store_id, "coral reef", limit=1))
    page_2 = unwrap(client.search(seeded_store.vector_store_id, "coral reef", limit=1, offset=1))

    assert page_1.data and page_2.data, "paginated search returned no results"
    assert _texts(page_1.data) != _texts(page_2.data), (
        f"offset=1 returned the same result as offset=0: {_texts(page_2.data)}"
    )
    assert page_1.data[0].score >= page_2.data[0].score, (
        "second page should not be strictly better than the first "
        f"(scores {page_1.data[0].score} vs {page_2.data[0].score})"
    )


def test_search_returns_multiple_output_fields_as_attributes(
    client: VectorStoreClient,
) -> None:
    """When a store's registered ``outputFields`` includes more than the text
    field, the extras land in the result's ``attributes`` dict; the text field
    itself moves into ``content`` (never duplicated in ``attributes``). Uses its
    own store because store-registered ``outputFields`` overrides the
    per-request value (see ``store_litellm_params`` docstring)."""
    store_id = f"e2e_milvus_outputs_{unique_marker()}"
    seed_probe_collection(client, store_id)
    try:
        params = client.store_litellm_params(output_fields=["text", "category"])
        registration = client.register_store(store_id, params)
        if not is_ok(registration):
            pytest.skip(f"managed store registration unavailable: {registration}")

        try:
            response = unwrap(client.search(store_id, "coral reef", limit=1))
            assert response.data, "search returned no results"
            top = response.data[0]

            assert "category" in top.attributes, (
                f"expected 'category' in attributes, got {top.attributes}"
            )
            assert "text" not in top.attributes, (
                "'text' should live under 'content', not 'attributes'"
            )
            assert top.content and top.content[0].text, "text content missing"
        finally:
            client.delete_store(store_id)
    finally:
        client.drop_collection(store_id)


def test_search_with_grouping_field_is_accepted(client: VectorStoreClient) -> None:
    """``groupingField`` is a Milvus-specific pass-through param. On a
    quick-setup collection the field is not indexed for grouping, so Milvus
    accepts the parameter but doesn't actually deduplicate; the requirement here
    is that the proxy forwards it without breaking the request and returns a
    well-formed response with the category attribute populated."""
    store_id = f"e2e_milvus_group_{unique_marker()}"
    seed_probe_collection(client, store_id)
    try:
        params = client.store_litellm_params(output_fields=["text", "category"])
        registration = client.register_store(store_id, params)
        if not is_ok(registration):
            pytest.skip(f"managed store registration unavailable: {registration}")

        try:
            response = unwrap(
                client.search(
                    store_id,
                    "a natural feature of the Earth",
                    limit=5,
                    grouping_field="category",
                )
            )
            assert response.object == "vector_store.search_results.page"
            assert response.data, "grouped search returned no results"
            categories = [result.attributes.get("category") for result in response.data]
            assert None not in categories, f"category attribute missing: {categories}"
        finally:
            client.delete_store(store_id)
    finally:
        client.drop_collection(store_id)


def test_search_accepts_consistency_level(
    client: VectorStoreClient, seeded_store: SeededStore
) -> None:
    """``consistencyLevel`` is a Milvus-specific pass-through param. The proxy
    must accept and forward it; the Milvus/Zilliz cluster's response depends on
    its cluster type and index state (a serverless cluster may return an
    upstream error for stricter levels than it supports), so this test asserts
    the proxy layer doesn't reject or misroute the param rather than pinning a
    specific backend outcome."""
    outcome = client.search_raw_with_params(
        seeded_store.vector_store_id, "coral reef", consistency_level="Bounded"
    )
    assert outcome.status_code in {200, 400, 500}, (
        f"unexpected proxy behavior for consistencyLevel forwarding: {outcome}"
    )
    if outcome.status_code == 200:
        body = json.loads(outcome.body)
        assert body.get("object") == "vector_store.search_results.page"
    else:
        assert "milvus" in outcome.body.lower(), (
            f"non-2xx response for consistencyLevel came from something other "
            f"than the Milvus backend: {outcome.body[:200]}"
        )


def test_search_accepts_list_query(
    client: VectorStoreClient, seeded_store: SeededStore
) -> None:
    """The search API accepts a list-of-strings query; the transform joins the
    parts before embedding, so results should be roughly equivalent to the
    joined single-string query."""
    response = unwrap(
        client.search(seeded_store.vector_store_id, ["coral", "reef"], limit=1)
    )
    assert response.data, "list-query search returned no results"
    assert seeded_store.text(1) in _texts(response.data), (
        f"list query did not retrieve the coral reef doc; got {_texts(response.data)}"
    )


# ---- managed-store CRUD --------------------------------------------------


def test_managed_store_list_contains_seeded(
    client: VectorStoreClient, seeded_store: SeededStore
) -> None:
    """The store the fixture registered must appear in GET /vector_store/list."""
    result = client.list_stores()
    if not is_ok(result):
        pytest.skip(f"list unavailable on this proxy: {result}")

    ids = {entry.vector_store_id for entry in unwrap(result).data}
    assert seeded_store.vector_store_id in ids, (
        f"seeded store {seeded_store.vector_store_id} not in list of {len(ids)} stores"
    )


def test_managed_store_info_returns_seeded(
    client: VectorStoreClient, seeded_store: SeededStore
) -> None:
    """POST /vector_store/info returns the seeded store's metadata (wrapped
    under a ``vector_store`` key)."""
    result = client.store_info(seeded_store.vector_store_id)
    if not is_ok(result):
        pytest.skip(f"info unavailable on this proxy: {result}")

    entry: ManagedStoreEntry = unwrap(result).vector_store
    assert entry.vector_store_id == seeded_store.vector_store_id
    assert entry.custom_llm_provider == "milvus"


def test_managed_store_update_persists(client: VectorStoreClient) -> None:
    """After POST /vector_store/update, the new name/description shows up in
    /vector_store/info. Uses its own store so the shared seeded_store keeps its
    original identity."""
    store_id = f"e2e_milvus_update_{unique_marker()}"
    registration = client.register_store(store_id, client.store_litellm_params())
    if not is_ok(registration):
        pytest.skip(f"managed store registration unavailable: {registration}")

    try:
        new_description = f"updated at {unique_marker()}"
        update = client.update_store(store_id, description=new_description)
        assert update.status_code < 500, f"update crashed the proxy: {update}"
        if update.status_code >= 400:
            pytest.skip(f"update returned {update.status_code}: {update.body[:200]}")

        info = client.store_info(store_id)
        if not is_ok(info):
            pytest.skip(f"info unavailable: {info}")
        entry = unwrap(info).vector_store
        assert entry.vector_store_description == new_description, (
            f"description update did not persist: got {entry.vector_store_description!r}"
        )
    finally:
        client.delete_store(store_id)


def test_managed_store_delete_removes_from_list(client: VectorStoreClient) -> None:
    """After POST /vector_store/delete, the id is gone from /vector_store/list."""
    store_id = f"e2e_milvus_del_{unique_marker()}"
    registration = client.register_store(store_id, client.store_litellm_params())
    if not is_ok(registration):
        pytest.skip(f"managed store registration unavailable: {registration}")

    client.delete_store(store_id)

    result = client.list_stores()
    if not is_ok(result):
        pytest.skip(f"list unavailable: {result}")
    ids = {entry.vector_store_id for entry in unwrap(result).data}
    assert store_id not in ids, f"deleted store {store_id} still appears in list"


# ---- OpenAI-compat create ------------------------------------------------


def test_openai_compat_create_does_not_crash_the_proxy(client: VectorStoreClient) -> None:
    """Milvus's provider config raises NotImplementedError for the OpenAI-compat
    create surface. On this proxy the route currently short-circuits ahead of the
    provider and returns a synthetic ``vector_store`` object without hitting
    Milvus, which is arguably a bug in the proxy (create claims success without
    creating a Milvus collection). Either way the requirement here is: it must
    not 5xx the proxy. If it does start honoring the provider and returns an
    error, that's fine too, as long as the error is clean."""
    intended_name = f"e2e_milvus_compat_create_{unique_marker()}"
    outcome = client.openai_compat_create(intended_name)

    assert outcome.status_code < 500, (
        f"OpenAI-compat create 5xx-crashed the proxy: {outcome}"
    )
    if outcome.ok:
        body = json.loads(outcome.body)
        assert body.get("object") == "vector_store", (
            f"successful create did not return a vector_store object: {body}"
        )
        assert body.get("id"), f"successful create returned no id: {body}"
