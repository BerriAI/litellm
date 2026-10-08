import httpx
import pytest
from pydantic import ValidationError

from litellm.llms.base_llm.search.transformation import SearchResponse, SearchResult
from litellm.llms.firecrawl.search.transformation import FirecrawlSearchConfig


def _transform(payload: object) -> SearchResponse:
    return FirecrawlSearchConfig().transform_search_response(
        raw_response=httpx.Response(200, json=payload), logging_obj=None
    )


def _entry(index: int) -> dict[str, str]:
    return {"title": f"Title {index}", "url": f"https://example.com/{index}", "description": f"Description {index}"}


def _news(index: int) -> dict[str, str]:
    return {"title": f"News {index}", "url": f"https://example.com/news/{index}", "snippet": f"Snippet {index}"}


def test_firecrawl_cloud_response_lists_web_results_before_news_results():
    response = _transform(
        {
            "success": True,
            "data": {
                "web": [
                    {**_entry(1), "markdown": "# Page 1", "metadata": {"statusCode": 200}},
                    {**_entry(2), "markdown": ""},
                    {**_entry(3), "markdown": None, "snippet": "ignored for web results"},
                    {"position": 4},
                ],
                "images": [{"title": "Logo", "imageUrl": "https://example.com/logo.png"}],
                "news": [
                    {**_news(1), "date": "2024-03-05", "markdown": "# News 1"},
                    {**_news(2), "date": None, "description": "ignored for news results"},
                    {"title": "News 3", "url": "https://example.com/news/3", "description": "ignored for news results"},
                ],
            },
            "creditsUsed": 2,
        }
    )

    assert response.object == "search"
    assert response.results == [
        SearchResult(title="Title 1", url="https://example.com/1", snippet="# Page 1"),
        SearchResult(title="Title 2", url="https://example.com/2", snippet="Description 2"),
        SearchResult(title="Title 3", url="https://example.com/3", snippet="Description 3"),
        SearchResult(title="", url="", snippet=""),
        SearchResult(title="News 1", url="https://example.com/news/1", snippet="# News 1", date="2024-03-05"),
        SearchResult(title="News 2", url="https://example.com/news/2", snippet="Snippet 2"),
        SearchResult(title="News 3", url="https://example.com/news/3", snippet="", date=None, last_updated=None),
    ]


def test_firecrawl_self_hosted_response_maps_the_flat_data_list():
    response = _transform(
        {
            "success": True,
            "data": [
                {**_entry(1), "markdown": "# Page 1"},
                {**_entry(2), "date": "2024-03-05"},
                {"metadata": {"statusCode": 200}},
            ],
        }
    )

    assert response.results == [
        SearchResult(title="Title 1", url="https://example.com/1", snippet="# Page 1"),
        SearchResult(title="Title 2", url="https://example.com/2", snippet="Description 2", date=None),
        SearchResult(title="", url="", snippet=""),
    ]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"success": False, "error": "rate limit"},
        {"data": None},
        {"data": "rate limit"},
        {"data": 429},
        {"data": []},
        {"data": {}},
        {"data": {"images": [{"imageUrl": "https://example.com/logo.png"}]}},
        {"data": {"web": [], "news": []}},
        {"data": {"web": "", "news": {}}},
    ],
)
def test_firecrawl_response_without_web_or_news_results_is_empty(payload):
    assert _transform(payload).results == []


@pytest.mark.parametrize("payload", [[_entry(1)], "upstream unavailable", True, 429])
def test_firecrawl_body_that_is_not_an_object_is_rejected_without_echoing_it(payload):
    with pytest.raises(ValidationError) as exc_info:
        _transform(payload)

    assert [error["type"] for error in exc_info.value.errors()] == ["dict_type"]
    assert "input_value" not in str(exc_info.value)


@pytest.mark.parametrize("section", ["web", "news"])
@pytest.mark.parametrize("value", [None, 429, True])
def test_firecrawl_section_that_cannot_be_iterated_is_rejected_without_echoing_it(section, value):
    with pytest.raises(ValidationError) as exc_info:
        _transform({"data": {"web": [_entry(1)], "news": [_news(1)], section: value}})

    assert [error["type"] for error in exc_info.value.errors()] == ["iterable_type"]
    assert "input_value" not in str(exc_info.value)


def _payload_with_entries(shape: str, entries: list[object]) -> dict[str, object]:
    if shape == "self-hosted":
        return {"data": entries}
    if shape == "web":
        return {"data": {"web": entries}}
    return {"data": {"web": [_entry(0)], "news": entries}}


@pytest.mark.parametrize("shape", ["self-hosted", "web", "news"])
@pytest.mark.parametrize("position", [0, 403])
@pytest.mark.parametrize("entry", ["rate limit", None, 429, [_entry(1)]])
def test_firecrawl_entry_that_is_not_an_object_is_rejected_without_its_position(shape, position, entry):
    entries = [_entry(index) for index in range(position)] + [entry]

    with pytest.raises(ValidationError) as exc_info:
        _transform(_payload_with_entries(shape, entries))

    assert [(error["type"], error["loc"]) for error in exc_info.value.errors()] == [("dict_type", ())]
    assert "input_value" not in str(exc_info.value)


@pytest.mark.parametrize("shape", ["self-hosted", "web", "news"])
@pytest.mark.parametrize("position", [0, 403])
def test_firecrawl_entry_with_several_fields_of_wrong_type_reports_every_field(shape, position):
    entries = [_entry(index) for index in range(position)] + [
        {"title": 7, "url": None, "markdown": ["Request timed out"]}
    ]

    with pytest.raises(ValidationError) as exc_info:
        _transform(_payload_with_entries(shape, entries))

    assert [(error["loc"], error["input"]) for error in exc_info.value.errors()] == [
        (("title",), 7),
        (("url",), None),
        (("snippet",), ["Request timed out"]),
    ]
    assert "Request timed out" in str(exc_info.value)


def test_firecrawl_news_entry_with_non_string_date_reports_the_date_next_to_the_other_fields():
    with pytest.raises(ValidationError) as exc_info:
        _transform({"data": {"news": [{**_news(1), "title": 7, "date": 1709596800}]}})

    assert [(error["loc"], error["input"]) for error in exc_info.value.errors()] == [
        (("title",), 7),
        (("date",), 1709596800),
    ]
