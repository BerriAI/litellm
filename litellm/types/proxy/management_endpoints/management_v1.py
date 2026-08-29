"""Shared response shapes for the `/management/v1` control-plane surface."""

from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

TOut = TypeVar("TOut")


class ProblemDetail(BaseModel):
    """RFC 9457 problem details, served as `application/problem+json`."""

    type: str
    title: str
    status: int
    detail: str
    allowed: list[str] | None = None


class PageLinks(BaseModel):
    """Hypermedia for a paginated list. No `first`/`last`: without a total count the last page is unknown."""

    model_config = ConfigDict(populate_by_name=True)

    self_link: str = Field(alias="self")
    prev: str | None = None
    next: str | None = None


class PageMeta(BaseModel):
    """`has_more` rather than `total_count`, which would need a COUNT(*) over the whole match set per keystroke."""

    page: int
    page_size: int
    has_more: bool


class FacetListResponse(BaseModel):
    """The distinct values one column takes over a filtered query. `data` holds bare values, not entity rows."""

    data: list[str]
    meta: PageMeta
    links: PageLinks


class ListMeta(BaseModel):
    """Page-mode counterpart to `PageMeta`: an entity list pays for the COUNT(*) so the table can show a page count."""

    total_count: int
    page: int
    page_size: int
    total_pages: int


class ListLinks(BaseModel):
    """Page-mode counterpart to `PageLinks`. `first`/`last` are knowable here because the total count is."""

    model_config = ConfigDict(populate_by_name=True)

    self_link: str = Field(alias="self")
    first: str
    prev: str | None = None
    next: str | None = None
    last: str


class ListResponse(BaseModel, Generic[TOut]):
    """Rows stay flat: JSON:API's `{type, id, attributes}` wrapper is a deliberate deviation, so every
    dashboard column accessor would otherwise have to go through `.attributes`."""

    data: list[TOut]
    meta: ListMeta
    links: ListLinks
