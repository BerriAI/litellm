"""Exa /search request contract, https://exa.ai/docs/reference/search."""

from collections.abc import Mapping, Sequence
from typing import Literal, TypeAlias

from typing_extensions import ReadOnly, Required, TypedDict

ExaSearchMode: TypeAlias = Literal["instant", "fast", "auto", "deep-lite", "deep", "deep-reasoning"]
ExaPageSection: TypeAlias = Literal["header", "navigation", "banner", "body", "sidebar", "footer", "metadata"]


class ExaTextOptions(TypedDict, total=False):
    maxCharacters: ReadOnly[int | None]
    includeHtmlTags: ReadOnly[bool | None]
    verbosity: ReadOnly[Literal["compact", "standard", "full"] | None]
    includeSections: ReadOnly[Sequence[ExaPageSection] | None]
    excludeSections: ReadOnly[Sequence[ExaPageSection] | None]


class ExaHighlightsOptions(TypedDict, total=False):
    """numSentences and highlightsPerUrl are deprecated compatibility fields."""

    query: ReadOnly[str | None]
    verbosity: ReadOnly[Literal["low", "medium", "high"] | None]
    dynamic: ReadOnly[bool | None]
    maxCharacters: ReadOnly[int | None]
    numSentences: ReadOnly[int | None]
    highlightsPerUrl: ReadOnly[int | None]


class ExaSummaryOptions(TypedDict, total=False):
    query: ReadOnly[str | None]
    schema: ReadOnly[Mapping[str, object] | None]


class ExaExtrasOptions(TypedDict, total=False):
    links: ReadOnly[int | None]
    imageLinks: ReadOnly[int | None]
    richImageLinks: ReadOnly[int | None]
    richLinks: ReadOnly[int | None]
    codeBlocks: ReadOnly[int | None]


class ExaContextOptions(TypedDict, total=False):
    maxCharacters: ReadOnly[int]


class ExaContentsOptions(TypedDict, total=False):
    """context, livecrawl, and crawledBeforeDate are deprecated compatibility fields."""

    text: ReadOnly[bool | ExaTextOptions | None]
    highlights: ReadOnly[bool | ExaHighlightsOptions | None]
    summary: ReadOnly[ExaSummaryOptions | None]
    extras: ReadOnly[ExaExtrasOptions | None]
    livecrawlTimeout: ReadOnly[int | None]
    maxAgeHours: ReadOnly[int | None]
    snapshotAsOf: ReadOnly[str | None]
    subpages: ReadOnly[int | None]
    subpageTarget: ReadOnly[str | Sequence[str] | None]
    context: ReadOnly[bool | ExaContextOptions | None]
    livecrawl: ReadOnly[Literal["never", "always", "fallback", "preferred"] | None]
    crawledBeforeDate: ReadOnly[str | None]


class ExaCountryLocation(TypedDict):
    country: ReadOnly[str]


class ExaCoordinateLocation(TypedDict, total=False):
    country: ReadOnly[str]
    latitude: Required[ReadOnly[float]]
    longitude: Required[ReadOnly[float]]


class ExaAISearchRequest(TypedDict, total=False):
    """Crawl-date filters, includeText, excludeText, and context are deprecated."""

    query: Required[ReadOnly[str]]
    type: ReadOnly[ExaSearchMode | None]
    category: ReadOnly[str | None]
    additionalQueries: ReadOnly[Sequence[str] | None]
    userLocation: ReadOnly[str | ExaCountryLocation | ExaCoordinateLocation | None]
    numResults: ReadOnly[int | None]
    includeDomains: ReadOnly[Sequence[str] | None]
    excludeDomains: ReadOnly[Sequence[str] | None]
    startPublishedDate: ReadOnly[str | None]
    endPublishedDate: ReadOnly[str | None]
    moderation: ReadOnly[bool | None]
    contents: ReadOnly[ExaContentsOptions | None]
    outputSchema: ReadOnly[Mapping[str, object] | None]
    objective: ReadOnly[str | None]
    systemPrompt: ReadOnly[str | None]
    userIdentifier: ReadOnly[str | None]
    compliance: ReadOnly[Literal["hipaa"] | None]
    stream: ReadOnly[bool | None]
    startCrawlDate: ReadOnly[str | None]
    endCrawlDate: ReadOnly[str | None]
    includeText: ReadOnly[Sequence[str] | None]
    excludeText: ReadOnly[Sequence[str] | None]
    context: ReadOnly[bool | ExaContextOptions | None]
