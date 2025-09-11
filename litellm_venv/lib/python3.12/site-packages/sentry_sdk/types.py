"""
This module contains type definitions for the Sentry SDK's public API.
The types are re-exported from the internal module `sentry_sdk._types`.

Disclaimer: Since types are a form of documentation, type definitions
may change in minor releases. Removing a type would be considered a
breaking change, and so we will only remove type definitions in major
releases.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentry_sdk._types import Event, EventDataCategory, Hint
else:
    from typing import Any

    # The lines below allow the types to be imported from outside `if TYPE_CHECKING`
    # guards. The types in this module are only intended to be used for type hints.
    Event = Any
    EventDataCategory = Any
    Hint = Any

__all__ = ("Event", "EventDataCategory", "Hint")
