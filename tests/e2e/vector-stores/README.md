# Vector store e2e

Live end-to-end coverage for litellm managed vector stores, driven through a
running proxy exactly the way a user would. The first (and currently only)
backend is Milvus / Zilliz Cloud

The suite stands up a real store and proves the whole path in one flow: it creates
a Milvus collection, embeds and inserts five known documents, registers the
collection as a managed vector store on the proxy, searches it over the
OpenAI-compatible route, and uses it as retrieval context in a chat completion.
One seeded document holds a made-up access code that appears nowhere else, so a
chat answer that echoes the code can only have come from the store being searched
and its content injected as context; that is the retrieval proof

## What it covers

- search returns the seeded documents in the OpenAI `vector_store.search_results.page` shape
- `limit` caps the number of results
- a Milvus `filter` expression restricts results to specific primary ids
- chat completions with `vector_store_ids` retrieve context and the answer echoes the seeded code
- a store pointing at a non-existent collection fails cleanly (non-2xx) instead of hanging or 5xx-crashing

## Credentials

Three credentials, nothing else to tune. Set them in the environment (or `.env`)
of the test process (and the proxy needs its own working provider keys)

| Variable | Notes |
| --- | --- |
| `MILVUS_API_BASE` | Zilliz Cloud cluster public endpoint, e.g. `https://in03-xxxx.serverless.gcp-us-west1.cloud.zilliz.com`. For self-hosted Milvus, the base URL of the REST server |
| `MILVUS_API_KEY` | Zilliz Cloud API key. For self-hosted Milvus, a `user:password` token |
| `OPENAI_API_KEY` | Used to embed the seed documents, and on the proxy to embed queries at search time and to back the chat model the suite registers |

Everything else is fixed in `milvus_client.py`: the embedding model
(`text-embedding-3-small`, 1536 dims) and the chat model. Rather than depend on
whatever chat model the proxy has, the suite registers its own deployment through
`POST /model/new` and deletes it on teardown, so the chat test is deterministic

The proxy must be reachable (see `tests/e2e/e2e_config.py` for `LITELLM_PROXY_URL`
/ `LITELLM_MASTER_KEY`), connected to a database (managed vector store
registration persists to the store registry), and started with
`STORE_MODEL_IN_DB=True` (so the chat model can be registered). Without model
storage, only the chat test skips; the search tests still run

Absent any of the required credentials, or a live proxy, the whole suite skips;
it never silently passes

## Running

```bash
cd tests/e2e
uv run --python 3.12 pytest vector-stores/ -v
```

The e2e harness uses `type` alias syntax, so Python 3.12+ is required
