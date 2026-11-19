import argparse
import os
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import accumulate, repeat, takewhile
from typing import Final, Literal, TextIO

import httpx
from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from litellm.vector_stores.utils import is_litellm_written_metadata_key

DEFAULT_API_BASE: Final = "https://api.openai.com/v1"
LIST_PAGE_SIZE: Final = "100"


class VectorStore(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    id: str
    metadata: dict[str, JsonValue] | None = None


class VectorStorePage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    data: tuple[VectorStore, ...]
    has_more: bool = False


class OpenAIErrorBody(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    message: str = ""


class OpenAIErrorResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    error: OpenAIErrorBody


@dataclass(frozen=True, slots=True)
class Options:
    api_key: str
    api_base: str
    vector_store_ids: tuple[str, ...]
    extra_keys: frozenset[str]
    apply: bool


@dataclass(frozen=True, slots=True)
class StoreResult:
    store_id: str
    matched_keys: tuple[str, ...]
    status: Literal["dry_run", "updated", "failed"]
    error: str = ""


def is_proxy_written_key(key: str, extra_keys: frozenset[str]) -> bool:
    return is_litellm_written_metadata_key(key) or key in extra_keys


def matching_keys(store: VectorStore, extra_keys: frozenset[str]) -> tuple[str, ...]:
    return tuple(sorted(key for key in (store.metadata or {}) if is_proxy_written_key(key, extra_keys)))


def fetch_page(client: httpx.Client, after: str | None) -> VectorStorePage:
    params: Final = {"limit": LIST_PAGE_SIZE} if after is None else {"limit": LIST_PAGE_SIZE, "after": after}
    return VectorStorePage.model_validate_json(client.get("/vector_stores", params=params).raise_for_status().content)


def fetch_next_page(client: httpx.Client, page: VectorStorePage | None) -> VectorStorePage | None:
    if page is None or not page.has_more or not page.data:
        return None
    return fetch_page(client, after=page.data[-1].id)


def list_all_stores(client: httpx.Client) -> Iterator[VectorStore]:
    pages: Final = accumulate(
        repeat(None), lambda page, _: fetch_next_page(client, page), initial=fetch_page(client, after=None)
    )
    for page in takewhile(lambda page: page is not None, pages):
        yield from page.data


def retrieve_stores(client: httpx.Client, store_ids: tuple[str, ...]) -> Iterator[VectorStore]:
    for store_id in store_ids:
        yield VectorStore.model_validate_json(client.get(f"/vector_stores/{store_id}").raise_for_status().content)


def describe_http_error(response: httpx.Response) -> str:
    try:
        message: Final = OpenAIErrorResponse.model_validate_json(response.content).error.message
    except ValidationError:
        return f"HTTP {response.status_code}"
    return f"HTTP {response.status_code}: {message}"


def delete_keys(client: httpx.Client, store_id: str, keys: tuple[str, ...]) -> StoreResult:
    try:
        response: Final = client.post(f"/vector_stores/{store_id}", json={"metadata": dict.fromkeys(keys)})
    except httpx.TransportError as error:
        return StoreResult(store_id=store_id, matched_keys=keys, status="failed", error=type(error).__name__)
    if response.is_success:
        return StoreResult(store_id=store_id, matched_keys=keys, status="updated")
    return StoreResult(store_id=store_id, matched_keys=keys, status="failed", error=describe_http_error(response))


def clean_store(client: httpx.Client, store: VectorStore, options: Options) -> StoreResult | None:
    keys: Final = matching_keys(store, options.extra_keys)
    if not keys:
        return None
    if not options.apply:
        return StoreResult(store_id=store.id, matched_keys=keys, status="dry_run")
    return delete_keys(client, store.id, keys)


def format_result(result: StoreResult) -> str:
    key_list: Final = ", ".join(result.matched_keys)
    count: Final = len(result.matched_keys)
    match result.status:
        case "dry_run":
            return f"{result.store_id}: would delete {count} keys: {key_list}"
        case "updated":
            return f"{result.store_id}: deleted {count} keys: {key_list}"
        case "failed":
            return f"{result.store_id}: failed to delete {count} keys ({result.error}): {key_list}"


def clean_and_report(
    client: httpx.Client, stores: Iterator[VectorStore], options: Options, out: TextIO
) -> Iterator[StoreResult | None]:
    for store in stores:
        result: Final = clean_store(client, store, options)
        if result is not None:
            print(format_result(result), file=out)
        yield result


def run(options: Options, client: httpx.Client, out: TextIO) -> int:
    stores: Final = (
        retrieve_stores(client, options.vector_store_ids) if options.vector_store_ids else list_all_stores(client)
    )
    scanned: Final = tuple(clean_and_report(client, stores, options, out))
    results: Final = tuple(result for result in scanned if result is not None)
    updated: Final = sum(1 for result in results if result.status == "updated")
    failed: Final = sum(1 for result in results if result.status == "failed")
    mode: Final = "applied" if options.apply else "dry run, pass --apply to delete"
    print(
        f"scanned={len(scanned)} matched={len(results)} updated={updated} failed={failed} ({mode})",
        file=out,
    )
    return 1 if failed else 0


def parse_options(argv: Sequence[str] | None) -> Options:
    parser: Final = argparse.ArgumentParser(
        description="Remove LiteLLM proxy-written keys from OpenAI vector store metadata. Dry run by default."
    )
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"), help="defaults to $OPENAI_API_KEY")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--vector-store-id", action="append", default=[], help="repeatable, defaults to every store")
    parser.add_argument("--extra-key", action="append", default=[], help="repeatable, another key name to delete")
    parser.add_argument("--apply", action="store_true", help="delete the matching keys instead of listing them")
    args: Final = parser.parse_args(argv)
    if not args.api_key:
        parser.error("--api-key or OPENAI_API_KEY is required")
    return Options(
        api_key=args.api_key,
        api_base=args.api_base.rstrip("/"),
        vector_store_ids=tuple(args.vector_store_id),
        extra_keys=frozenset(args.extra_key),
        apply=args.apply,
    )


def main(argv: Sequence[str] | None = None, transport: httpx.BaseTransport | None = None) -> int:
    options: Final = parse_options(argv)
    with httpx.Client(
        base_url=options.api_base,
        headers={"Authorization": f"Bearer {options.api_key}"},
        timeout=60.0,
        transport=transport,
    ) as client:
        try:
            return run(options, client, sys.stdout)
        except httpx.HTTPStatusError as error:
            sys.stderr.write(f"could not read vector stores: {describe_http_error(error.response)}\n")
            return 2
        except httpx.TransportError as error:
            sys.stderr.write(f"could not reach {options.api_base}: {type(error).__name__}\n")
            return 2
        except ValidationError:
            sys.stderr.write(f"unexpected vector store response shape from {options.api_base}\n")
            return 2


if __name__ == "__main__":
    sys.exit(main())
