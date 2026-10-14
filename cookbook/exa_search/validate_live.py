"""Run explicitly opted-in Exa checks, writing responses without credentials."""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit

import jsonschema

import litellm
from litellm.llms.base_llm.search.streaming_iterator import SearchStreamWrapper

QUERY: Final = "What does the Python documentation say about asyncio TaskGroup?"
SCHEMA: Final = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}


def cases():
    for mode in ("instant", "fast", "auto", "deep-lite", "deep", "deep-reasoning"):
        yield f"{mode}-highlights", {"type": mode, "contents": {"highlights": True}}
    yield (
        "dynamic-highlights",
        {
            "contents": {"highlights": {"dynamic": True, "verbosity": "low"}},
            "extra_headers": {"Exa-Beta": "dynamic-highlights-2026-08-28"},
        },
    )
    yield (
        "text-options",
        {
            "contents": {
                "text": {"maxCharacters": 1200, "verbosity": "compact", "includeSections": ["body"]},
                "maxAgeHours": 0,
            }
        },
    )
    yield "summary-options", {"contents": {"summary": {"query": "Explain TaskGroup"}}}
    yield "metadata-only", {"contents": {}}
    yield (
        "object-output",
        {"contents": {"highlights": True}, "outputSchema": SCHEMA, "systemPrompt": "Ground the answer in the sources."},
    )
    yield "text-output", {"contents": {"highlights": True}, "outputSchema": {"type": "text"}}
    yield (
        "deep-additional-queries",
        {
            "type": "deep",
            "additionalQueries": ["Python TaskGroup exception handling"],
            "contents": {"highlights": True},
        },
    )
    yield "location-object", {"userLocation": {"country": "US"}, "contents": {"highlights": True}}
    yield (
        "extras",
        {
            "contents": {
                "text": {"maxCharacters": 500},
                "extras": {"links": 2, "richLinks": 2, "imageLinks": 2, "richImageLinks": 2, "codeBlocks": 2},
            }
        },
    )
    yield "snapshot", {"contents": {"highlights": True, "snapshotAsOf": "2026-01-01"}}
    yield "stream-text", {"stream": True, "outputSchema": {"type": "text"}, "contents": {"highlights": True}}
    yield "stream-object", {"stream": True, "outputSchema": SCHEMA, "contents": {"highlights": True}}
    yield "stream-json-fallback", {"stream": True, "contents": {"highlights": True}}
    yield "legacy-default-text", {}
    yield "null-contents", {"contents": None}
    yield "highlight-options", {"contents": {"highlights": {"query": "TaskGroup exceptions", "maxCharacters": 600}}}
    yield "summary-schema", {"contents": {"summary": {"query": "Explain TaskGroup", "schema": SCHEMA}}}
    yield "location-coordinates", {"userLocation": {"latitude": 40.7, "longitude": -74.0}, "contents": {}}
    yield "domain-filter", {"includeDomains": ["docs.python.org"], "contents": {"highlights": True}}
    yield "subpages", {"contents": {"highlights": True, "subpages": 1, "subpageTarget": ["asyncio"]}}
    yield (
        "text-html-sections",
        {
            "contents": {
                "text": {"includeHtmlTags": True, "excludeSections": ["navigation", "footer"], "maxCharacters": 600},
                "maxAgeHours": 0,
            }
        },
    )

    for mode in ("instant", "fast", "auto", "deep-lite", "deep", "deep-reasoning"):
        yield (
            f"{mode}-stream-object",
            {"type": mode, "stream": True, "outputSchema": SCHEMA, "contents": {"highlights": True}},
        )

    yield (
        "subpages-homepage",
        {
            "query": "Python official documentation homepage",
            "includeDomains": ["docs.python.org"],
            "contents": {"text": {"maxCharacters": 500}, "subpages": 2, "subpageTarget": ["library"]},
        },
    )
    yield (
        "image-extras",
        {
            "query": "Wikimedia Commons cat photographs",
            "includeDomains": ["commons.wikimedia.org"],
            "contents": {"text": {"maxCharacters": 500}, "extras": {"imageLinks": 2, "richImageLinks": 2}},
        },
    )
    yield (
        "native-controls",
        {
            "query": "Research papers on structured concurrency",
            "type": "fast",
            "category": "publication",
            "moderation": True,
            "startPublishedDate": "2000-01-01T00:00:00Z",
            "endPublishedDate": "2027-01-01T00:00:00Z",
            "excludeDomains": ["example.com"],
            "objective": "Find research evidence on structured concurrency",
            "userIdentifier": "litellm-validation",
            "contents": {"highlights": True, "maxAgeHours": -1, "livecrawlTimeout": 5000},
        },
    )


async def run(asynchronous: bool, parameters):
    response = await litellm.asearch(**parameters) if asynchronous else litellm.search(**parameters)
    if isinstance(response, SearchStreamWrapper):
        if asynchronous:
            async with response:
                events = [event.model_dump() async for event in response]
        else:
            with response:
                events = [event.model_dump() for event in response]
        assert events[-1]["type"] == "done"
        assert response.final_response is not None
        return response.final_response.model_dump(), events
    return response.model_dump(), []


async def main(output: Path, selected: list[str]):
    key = os.environ.get("EXA_API_KEY")
    if not key:
        raise SystemExit("EXA_API_KEY is unavailable. No live requests were made.")
    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for index, (name, params) in enumerate(cases()):
        if selected and name not in selected:
            continue
        started = time.monotonic()
        asynchronous = index % 2 == 0
        try:
            response, events = await run(
                asynchronous, {"query": QUERY, "search_provider": "exa_ai", "numResults": 2, "timeout": 120, **params}
            )
            if params.get("stream") and params.get("outputSchema") is not None:
                assert events, "Requested SSE but received ordinary JSON"
            assert isinstance(response["results"], list)
            if "requestId" in response:
                assert isinstance(response["requestId"], str)
            if "searchTime" in response:
                assert isinstance(response["searchTime"], (int, float))
            if "costDollars" in response:
                assert isinstance(response["costDollars"], dict)
            if params.get("outputSchema") == SCHEMA:
                jsonschema.validate(response["output"]["content"], SCHEMA)
                assert isinstance(response["output"]["grounding"], list)
            if name.endswith("highlights"):
                assert any(result.get("highlights") for result in response["results"])
                assert all(not result.get("text") for result in response["results"])
            if name in ("text-options", "legacy-default-text", "text-html-sections"):
                assert any(result.get("text") for result in response["results"])
            if name in ("summary-options", "summary-schema"):
                assert any(result.get("summary") for result in response["results"])
            if name == "summary-schema":
                for result in response["results"]:
                    if result.get("summary"):
                        jsonschema.validate(json.loads(result["summary"]), SCHEMA)
            if params.get("outputSchema", {}).get("type") == "text":
                assert isinstance(response["output"]["content"], str)
                assert isinstance(response["output"]["grounding"], list)
            if name == "domain-filter":
                assert response["results"]
                assert all(urlsplit(result["url"]).hostname == "docs.python.org" for result in response["results"])
            if name in ("null-contents", "metadata-only"):
                assert all(not result.get("text") for result in response["results"])
            records.append(
                {
                    "case": name,
                    "async": asynchronous,
                    "parameters": params,
                    "status": "passed",
                    "seconds": time.monotonic() - started,
                    "response": response,
                    "events": events,
                }
            )
        except Exception as error:
            records.append(
                {
                    "case": name,
                    "async": asynchronous,
                    "parameters": params,
                    "status": "failed",
                    "seconds": time.monotonic() - started,
                    "error": repr(error).replace(key, "[REDACTED]"),
                }
            )
        output.write_text(json.dumps(records, indent=2))
        print(name, records[-1]["status"], flush=True)
    if any(record["status"] != "passed" for record in records):
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", default=[])
    args = parser.parse_args()
    asyncio.run(main(args.output, args.case))
