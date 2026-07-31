"""Shared response shapes for the `/management/v1` control-plane surface."""

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field, JsonValue


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
    """An entity list can afford the COUNT(*) a facet cannot, so it reports a real total."""

    page: int
    page_size: int
    total_count: int
    total_pages: int


class ListLinks(BaseModel):
    """Hypermedia for an entity list. `first`/`last` exist here because `total_pages` is known."""

    model_config = ConfigDict(populate_by_name=True)

    self_link: str = Field(alias="self")
    first: str
    prev: str | None = None
    next: str | None = None
    last: str


class ListResponse(BaseModel):
    """One page of an entity collection. Rows are flat: no `{type, id, attributes}` wrapper."""

    data: tuple[Mapping[str, JsonValue], ...]
    meta: ListMeta
    links: ListLinks
