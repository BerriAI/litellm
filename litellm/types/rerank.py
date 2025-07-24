"""
LiteLLM Follows the cohere API format for the re rank API
https://docs.cohere.com/reference/rerank

"""

from typing import List, Optional, Union

from pydantic import BaseModel, PrivateAttr
from typing_extensions import Required, TypedDict


class RerankRequest(BaseModel):
    model: str
    query: str
    top_n: Optional[int] = None
    documents: List[Union[str, dict]]
    rank_fields: Optional[List[str]] = None
    return_documents: Optional[bool] = None
    max_chunks_per_doc: Optional[int] = None
    max_tokens_per_doc: Optional[int] = None


class OptionalRerankParams(TypedDict, total=False):
    query: str
    top_n: Optional[int]
    documents: List[Union[str, dict]]
    rank_fields: Optional[List[str]]
    return_documents: Optional[bool]
    max_chunks_per_doc: Optional[int]
    max_tokens_per_doc: Optional[int]


class RerankBilledUnits(TypedDict, total=False):
    search_units: Optional[int]
    total_tokens: Optional[int]


class RerankTokens(TypedDict, total=False):
    input_tokens: Optional[int]
    output_tokens: Optional[int]


class RerankResponseMeta(TypedDict, total=False):
    api_version: Optional[dict]
    billed_units: Optional[RerankBilledUnits]
    tokens: Optional[RerankTokens]


class RerankResponseDocument(TypedDict):
    text: str


class RerankResponseResult(TypedDict, total=False):
    index: Required[int]
    relevance_score: Required[float]
    document: RerankResponseDocument


class RerankResponse(BaseModel):
    """
    Rerank response model.

    Note: This class provides a virtual 'usage' property that doesn't exist in the actual
    rerank API responses but is required for LiteLLM's TPM/RPM tracking. The usage data
    is computed from the meta.tokens or meta.billed_units fields when available.
    """

    id: Optional[str] = None
    results: Optional[List[RerankResponseResult]] = None  # Contains index and relevance_score
    meta: Optional[RerankResponseMeta] = None  # Contains api_version and billed_units

    # Define private attributes using PrivateAttr
    _hidden_params: dict = PrivateAttr(default_factory=dict)

    @property
    def usage(self) -> dict:
        """Virtual property for compatibility with TPM/RPM tracking."""
        return self._get_usage_dict()

    def __getitem__(self, key):
        # Handle usage key specially
        if key == "usage":
            return self._get_usage_dict()
        return self.__dict__[key]

    def get(self, key, default=None):
        # Handle usage key specially
        if key == "usage":
            return self._get_usage_dict()
        return self.__dict__.get(key, default)

    def __contains__(self, key):
        # Handle usage key specially
        if key == "usage":
            return True
        return key in self.__dict__

    def _get_usage_dict(self) -> dict:
        """
        Create a usage dict from the rerank response metadata.
        This allows rerank responses to work with TPM/RPM tracking.
        """
        # Handle meta being None
        if not self.meta:
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 1}

        # Safely get tokens - handle both dict and pydantic model cases
        tokens = getattr(self.meta, "tokens", None)
        if tokens and hasattr(tokens, "model_dump"):
            # It's a pydantic model
            tokens = tokens.model_dump()
        elif tokens and not isinstance(tokens, dict):
            # Convert to dict if it's some other type
            tokens = dict(tokens) if hasattr(tokens, "__dict__") else {}

        # Safely get billed_units - handle both dict and pydantic model cases
        billed_units = getattr(self.meta, "billed_units", None)
        if billed_units and hasattr(billed_units, "model_dump"):
            # It's a pydantic model
            billed_units = billed_units.model_dump()
        elif billed_units and not isinstance(billed_units, dict):
            # Convert to dict if it's some other type
            billed_units = dict(billed_units) if hasattr(billed_units, "__dict__") else {}

        # Get token counts with safe defaults
        input_tokens = (tokens or {}).get("input_tokens", 0) or 0
        output_tokens = (tokens or {}).get("output_tokens", 0) or 0
        total_tokens = input_tokens + output_tokens

        # If no token info but we have billed_units, use that
        if total_tokens == 0 and billed_units:
            total_tokens = billed_units.get("total_tokens", 0) or 0

        # Default to 1 token for tracking purposes if still 0
        if total_tokens == 0:
            total_tokens = 1

        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": total_tokens}

    def model_dump(self, **kwargs):
        """Override model_dump to include usage field."""
        data = super().model_dump(**kwargs)
        data["usage"] = self._get_usage_dict()
        return data
