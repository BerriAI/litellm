"""
Default parameter values and shared constants for APISerpent search.

Single source of truth for the supported request parameters and their
package-level defaults. See https://apiserpent.com/docs.
"""

from dataclasses import asdict, dataclass
from typing import Dict, Literal, Optional

SearchEngine = Literal["google", "bing", "yahoo", "ddg"]
SafeSearch = Literal["off", "moderate", "strict"]
Freshness = Literal["h", "1h", "d", "1d", "7d", "w", "m", "1m", "y", "1y"]
ResponseFormat = Literal["full", "simple"]

NUM_MIN = 1
NUM_MIN_DEEP = 10
NUM_MAX = 100
PAGES_MIN = 1
PAGES_MAX = 10


@dataclass(frozen=True)
class APISerpentSearchParams:
    """
    Supported APISerpent search parameters with package defaults.

    Fields defaulting to ``None`` are only sent when the caller provides them;
    the rest are always sent so behavior is deterministic regardless of any
    server-side defaults.
    """

    engine: SearchEngine = "google"
    country: str = "us"
    num: int = 10
    format: ResponseFormat = "full"
    pages: Optional[int] = None
    freshness: Optional[Freshness] = None
    safe: Optional[SafeSearch] = None
    language: Optional[str] = None
    pixel_position: Optional[bool] = None

    def __post_init__(self) -> None:
        # num's deep-search floor (NUM_MIN_DEEP) is endpoint-specific and enforced
        # in the transform layer; here we only bound the absolute range.
        if not NUM_MIN <= self.num <= NUM_MAX:
            raise ValueError(
                f"num must be between {NUM_MIN} and {NUM_MAX}, got {self.num}"
            )
        if self.pages is not None and not PAGES_MIN <= self.pages <= PAGES_MAX:
            raise ValueError(
                f"pages must be between {PAGES_MIN} and {PAGES_MAX}, got {self.pages}"
            )

    def to_request_params(self) -> Dict:
        """Return non-None fields as request params, booleans lowercased."""
        params: Dict = {}
        for key, value in asdict(self).items():
            if value is None:
                continue
            params[key] = str(value).lower() if isinstance(value, bool) else value
        return params

    @classmethod
    def field_names(cls) -> set:
        return set(cls.__dataclass_fields__.keys())


QUICK_SEARCH_PATH = "/api/search/quick"
DEEP_SEARCH_PATH = "/api/search"
