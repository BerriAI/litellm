"""
Used for /watsonx/ pass-through endpoints.
"""

from typing_extensions import NotRequired, ReadOnly, TypedDict


class WatsonxPassThroughCredentials(TypedDict):
    api_key: ReadOnly[NotRequired[str | None]]
    api_base: ReadOnly[NotRequired[str | None]]
    project_id: ReadOnly[NotRequired[str | None]]
    space_id: ReadOnly[NotRequired[str | None]]
    model: ReadOnly[NotRequired[str | None]]
