"""
Memory snapshot capture and storage system.

This package provides functionality for:
- Capturing memory snapshots from tracemalloc
- Smart capture decision logic (detailed vs lightweight snapshots)
- Storing snapshots to JSON files
- Retrieving snapshot file statistics
"""

from .capture import (
    should_capture_detailed_snapshot,
    capture_request_memory_snapshot,
)
from .storage import (
    sanitize_filename,
    save_buffered_snapshots_to_json,
    get_snapshot_file_info,
    save_batch_snapshots,
    print_final_snapshot_summary,
)

__all__ = [
    # Capture functions
    "should_capture_detailed_snapshot",
    "capture_request_memory_snapshot",
    # Storage functions
    "sanitize_filename",
    "save_buffered_snapshots_to_json",
    "get_snapshot_file_info",
    "save_batch_snapshots",
    "print_final_snapshot_summary",
]

