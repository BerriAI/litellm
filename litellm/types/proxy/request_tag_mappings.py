from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self


class RequestTagMapping(BaseModel):
    """An exact baggage value mapping, or a prefix for any value of a key."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    header: Literal["baggage"] = "baggage"
    key: str = Field(min_length=1, pattern=r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")
    value: str | None = Field(None, min_length=1)
    tag: str | None = Field(None, min_length=1)
    tag_prefix: str | None = Field(None, min_length=1)
    required: bool = False
    require_budgeted_tag: bool = False

    @model_validator(mode="after")
    def validate_mapping(self) -> Self:
        if self.value is not None:
            if self.tag is None or self.tag_prefix is not None:
                raise ValueError("An exact value mapping requires tag and cannot set tag_prefix")
        elif self.tag_prefix is None or self.tag is not None:
            raise ValueError("An any-value mapping requires tag_prefix and cannot set tag")
        return self
