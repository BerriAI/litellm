"""Shared response shapes for the `/management/v1` control-plane surface."""

from pydantic import BaseModel, ConfigDict, Field


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
