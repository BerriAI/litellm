"""
JSON utilities for caching implementations.

This module provides shared JSON encoding functionality across all cache implementations.
"""

import json
from datetime import timedelta
from typing import Any


class TimedeltaJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles timedelta objects by converting them to seconds.

    This encoder is used across all cache implementations (Redis, S3, GCS, Azure Blob)
    to prevent 'Object of type timedelta is not JSON serializable' errors when
    caching metrics that contain timedelta objects.

    Example:
        >>> import json
        >>> from datetime import timedelta
        >>> data = {"latency": [timedelta(seconds=1.5)]}
        >>> json.dumps(data, cls=TimedeltaJSONEncoder)
        '{"latency": [1.5]}'
    """

    def default(self, obj: Any) -> Any:
        """
        Convert timedelta objects to seconds (float) for JSON serialization.

        Args:
            obj: Object to serialize

        Returns:
            Serializable representation of the object
        """
        if isinstance(obj, timedelta):
            return obj.total_seconds()
        return super().default(obj)
