from typing import Dict, Literal, Mapping, Optional, Tuple, Union

from fastapi import HTTPException
from pydantic import BaseModel, Field, ValidationError

CONTAINER_LIST_QUERY_PARAMS: Tuple[str, ...] = ("after", "limit", "order")


class ContainerListPaginationParams(BaseModel):
    after: Optional[str] = None
    limit: Optional[int] = Field(default=None, ge=1)
    order: Optional[Literal["asc", "desc"]] = None


def parse_container_list_query_params(
    query_params: Mapping[str, str],
    supported_params: Tuple[str, ...] = CONTAINER_LIST_QUERY_PARAMS,
) -> Dict[str, Union[str, int]]:
    """
    Validate the pagination query params of a container list route and return them as
    top-level SDK arguments (`after`, `limit`, `order`), which is what the container
    SDK functions and the provider APIs expect.
    """
    requested = {key: value for key, value in query_params.items() if key in supported_params and value != ""}
    try:
        parsed = ContainerListPaginationParams.model_validate(requested)
    except ValidationError as e:
        violations = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}" for error in e.errors()
        )
        raise HTTPException(
            status_code=400,
            detail={"error": f"Invalid container list query parameters: {violations}"},
        )
    return parsed.model_dump(exclude_none=True)
