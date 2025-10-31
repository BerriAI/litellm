"""
Storage utilities for memory snapshots.

Provides functions for:
- Sanitizing filenames
- Saving/loading snapshot data to/from JSON files
- Getting snapshot file statistics
"""

import json
import os
import re
from typing import Dict, Any, List, Optional

from ..constants import DEFAULT_MAX_SNAPSHOTS_PER_TEST


def sanitize_filename(test_name: str) -> str:
    """
    Convert test name to a safe filename.
    
    Args:
        test_name: The test name to sanitize
        
    Returns:
        Safe filename string
        
    Example:
        >>> sanitize_filename("My Test: Async/Streaming")
        'my_test_async_streaming'
    """
    # Replace spaces and special characters with underscores
    safe_name = re.sub(r'[^\w\s-]', '', test_name)
    safe_name = re.sub(r'[-\s]+', '_', safe_name)
    return safe_name.strip('_').lower()


def save_buffered_snapshots_to_json(
    buffered_data: List[Dict[str, Any]],
    output_dir: str,
    test_name: str,
    max_snapshots_per_test: int = DEFAULT_MAX_SNAPSHOTS_PER_TEST
) -> None:
    """
    Write buffered memory snapshot data to a dedicated JSON file per test.
    
    Each test gets its own file: {output_dir}/{sanitized_test_name}.json
    File structure: [snapshot1, snapshot2, snapshot3, ...]
    
    This approach is much more efficient for multiple tests because:
    1. No file locking issues - tests can run in parallel
    2. Smaller files - only load/save data for one test
    3. Faster I/O - no need to read/write entire multi-test file
    4. Easier analysis - one file per test
    5. Easy cleanup - delete specific test files without affecting others
    
    Efficiently handles large files by:
    1. Capping requests per test to prevent unbounded growth
    2. Rotating out old requests when limit is exceeded
    3. Using compact JSON format (no indentation) to reduce file size
    
    Args:
        buffered_data: List of memory snapshot data to save (one snapshot per request)
        output_dir: Directory to store snapshot files
        test_name: Name of the test (used to create filename)
        max_snapshots_per_test: Maximum number of request snapshots per test (0 = unlimited)
        
    Example:
        >>> data = [{'request_id': 'req-1', 'current_mb': 10.5}]
        >>> save_buffered_snapshots_to_json(data, "memory_snapshots", "test_async")
    """
    if not buffered_data:
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create safe filename from test name
    safe_filename = sanitize_filename(test_name)
    output_file = os.path.join(output_dir, f"{safe_filename}.json")
    
    # Load existing data if file exists
    existing_data = []
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    print(f"[WARNING] Unexpected JSON format in {output_file}, creating new file")
                    existing_data = []
        except (json.JSONDecodeError, IOError) as e:
            print(f"[WARNING] Could not load existing JSON file {output_file}: {e}. Creating new file.")
            existing_data = []
    
    # Append buffered data
    existing_data.extend(buffered_data)
    
    # Rotate old requests if we exceed max_snapshots_per_test
    if max_snapshots_per_test > 0 and len(existing_data) > max_snapshots_per_test:
        requests_to_remove = len(existing_data) - max_snapshots_per_test
        existing_data = existing_data[-max_snapshots_per_test:]
        print(f"[INFO] Rotated out {requests_to_remove} old request snapshots for test '{test_name}', "
              f"keeping {max_snapshots_per_test} most recent requests")
    
    # Save to file using compact format (no indentation) to reduce file size
    try:
        with open(output_file, 'w') as f:
            json.dump(existing_data, f, separators=(',', ':'))
    except IOError as e:
        print(f"[ERROR] Could not save to JSON file {output_file}: {e}")


def get_snapshot_file_info(output_file: str) -> Optional[Dict[str, Any]]:
    """
    Get information about a snapshot JSON file for a specific test.
    
    Returns file statistics including:
    - Total requests captured in this test file
    - File size
    - Detailed vs lightweight snapshot counts
    - Date range of snapshots
    
    Note: Each entry = one request snapshot
    
    Args:
        output_file: Path to the snapshot JSON file (single test file)
        
    Returns:
        Dictionary with file info, or None if file doesn't exist or is invalid
        
    Example:
        >>> info = get_snapshot_file_info("memory_snapshots/test_async.json")
        >>> if info:
        ...     print(f"Total entries: {info['total_entries']}")
    """
    if not os.path.exists(output_file):
        return None
    
    try:
        file_size = os.path.getsize(output_file)
        
        with open(output_file, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            return None
        
        if not data:
            return {
                'file_size_mb': 0,
                'total_entries': 0,
                'detailed_entries': 0,
                'lightweight_entries': 0,
                'date_range': None
            }
        
        detailed_count = sum(1 for entry in data if 'top_consumers' in entry)
        lightweight_count = len(data) - detailed_count
        
        # Extract date range if timestamps exist
        timestamps = [entry.get('timestamp') for entry in data if 'timestamp' in entry]
        date_range = None
        if timestamps:
            date_range = {
                'first': timestamps[0],
                'last': timestamps[-1]
            }
        
        return {
            'file_size_mb': round(file_size / (1024 * 1024), 2),
            'total_entries': len(data),
            'detailed_entries': detailed_count,
            'lightweight_entries': lightweight_count,
            'date_range': date_range
        }
    except (json.JSONDecodeError, IOError) as e:
        print(f"[ERROR] Could not read snapshot file: {e}")
        return None


def save_batch_snapshots(
    snapshot_buffer: List[Dict[str, Any]],
    output_dir: str,
    test_name: str
) -> None:
    """
    Save buffered snapshots from one batch to JSON file.
    
    Writes snapshots after each batch to prevent memory buildup and data loss.
    This is more efficient than per-request writes while avoiding the memory
    overhead of holding all snapshots until the end.
    
    Args:
        snapshot_buffer: List of memory snapshot data to save (one batch worth)
        output_dir: Directory to store snapshot files
        test_name: Name of the test (used to create filename)
        
    Example:
        >>> buffer = [{'request_id': 'req-1', 'current_mb': 10.5}]
        >>> save_batch_snapshots(buffer, "memory_snapshots", "test_async")
    """
    if snapshot_buffer:
        save_buffered_snapshots_to_json(snapshot_buffer, output_dir, test_name)


def print_final_snapshot_summary(
    output_dir: str,
    test_name: str,
    smart_capture: bool
) -> None:
    """
    Print summary information about saved snapshots at the end of the test.
    
    Args:
        output_dir: Directory where snapshot files are stored
        test_name: Name of the test (used to find the file)
        smart_capture: Whether smart capture mode was enabled
        
    Example:
        >>> print_final_snapshot_summary("memory_snapshots", "test_async", True)
        [INFO] Memory snapshots for test 'test_async' saved to: ...
               Total requests captured: 1000 (50 detailed, 950 lightweight)
               File size: 2.5 MB
    """
    safe_filename = sanitize_filename(test_name)
    output_file = os.path.join(output_dir, f"{safe_filename}.json")
    
    # Get file info to show final statistics
    file_info = get_snapshot_file_info(output_file)
    
    if file_info:
        capture_info = ""
        if smart_capture:
            capture_info = f" ({file_info['detailed_entries']} detailed, {file_info['lightweight_entries']} lightweight)"
        
        print(f"\n[INFO] Memory snapshots for test '{test_name}' saved to: {output_file}")
        print(f"       Total requests captured: {file_info['total_entries']}{capture_info}")
        print(f"       File size: {file_info['file_size_mb']} MB")

