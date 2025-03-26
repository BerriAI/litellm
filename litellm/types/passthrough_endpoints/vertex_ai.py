"""
Used for /vertex_ai/ pass through endpoints
"""

from typing import Optional

from pydantic import BaseModel

from ..llms.vertex_ai import VERTEX_CREDENTIALS_TYPES


class VertexPassThroughCredentials(BaseModel):
    # Example: vertex_project = "my-project-123"
    vertex_project: Optional[str] = None

    # Example: vertex_location = "us-central1"
    vertex_location: Optional[str] = None

    # Example: vertex_credentials = "/path/to/credentials.json" or "os.environ/GOOGLE_CREDS"
    vertex_credentials: Optional[VERTEX_CREDENTIALS_TYPES] = None
