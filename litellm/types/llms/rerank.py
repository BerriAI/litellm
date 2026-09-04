from typing_extensions import (
    TypedDict,
)


class InfinityRerankResult(TypedDict):
    index: int
    relevance_score: float
    document: str | None
