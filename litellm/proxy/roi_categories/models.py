from datetime import datetime
from typing import Annotated, Final, Literal, TypeAlias

from pydantic import AfterValidator, AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from litellm.litellm_core_utils.secret_redaction import redact_string

Excerpt: TypeAlias = Annotated[str, AfterValidator(redact_string)]

UNCLASSIFIED: Final = "unclassified"
CONFIG_KEY: Final = "roi_categories"


class Category(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,47}$")
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=1000)


class Categories(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    categories: tuple[Category, ...] = Field(min_length=1, max_length=12)

    @model_validator(mode="after")
    def unique_categories(self) -> "Categories":
        identifiers: Final = tuple(category.id for category in self.categories)
        if len(set(identifiers)) != len(identifiers) or UNCLASSIFIED in identifiers:
            raise ValueError("Category IDs must be unique and cannot be unclassified")
        return self


class Setup(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    sample_size: int = Field(default=100, ge=5, le=500)
    discovery_model: str = Field(min_length=1, max_length=200)
    classifier_model: str = Field(min_length=1, max_length=200)
    context: Excerpt = Field(default="", max_length=32000)
    team_id: str = Field(default="", max_length=200)


class Settings(Setup):
    version: str
    started_at: AwareDatetime
    activated_at: AwareDatetime | None = None
    phase: Literal["sampling", "review", "active", "error"] = "sampling"
    categories: tuple[Category, ...] = ()
    sampled: int = 0
    error: str | None = None


class Status(BaseModel):
    model_config = ConfigDict(frozen=True)
    configured: bool
    settings: Settings | None = None


class Traffic(BaseModel):
    model_config = ConfigDict(frozen=True)
    request_id: str
    excerpt: Excerpt
    attempts: int = 0


class Assignment(BaseModel):
    model_config = ConfigDict(frozen=True)
    request_id: str
    version: str
    category_id: str = UNCLASSIFIED
    attempts: int
    completed: bool


class SpendBucket(BaseModel):
    model_config = ConfigDict(frozen=True)
    day: str
    category_id: str
    model: str
    spend: float
    requests: int
    unknown_cost_requests: int = 0


class Example(BaseModel):
    model_config = ConfigDict(frozen=True)
    request_id: str
    model: str
    spend: float | None
    excerpt: Excerpt


class Report(BaseModel):
    model_config = ConfigDict(frozen=True)
    categories: tuple[Category, ...]
    buckets: tuple[SpendBucket, ...]
    version: str


class ContextText(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    text: str


def utc_milliseconds(value: datetime) -> int:
    return int(value.timestamp() * 1000)


class RetryDiscovery(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ContextUpload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    name: str = Field(min_length=1, max_length=200)
    content_base64: str = Field(min_length=1, max_length=1398104)
