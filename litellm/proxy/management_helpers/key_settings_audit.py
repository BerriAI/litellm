"""Audit stamping for virtual key configuration changes."""

from collections.abc import Mapping
from datetime import datetime, timezone


def with_settings_updated_at(data: Mapping[str, object]) -> dict[str, object]:
    """Stamp a key update payload with the time its configuration changed.

    ``updated_at`` carries Prisma's ``@updatedAt`` and so is rewritten by every
    spend flush, which makes it useless for auditing; ``settings_updated_at`` is
    written only from key-management write paths.
    """
    return {**data, "settings_updated_at": datetime.now(timezone.utc)}
