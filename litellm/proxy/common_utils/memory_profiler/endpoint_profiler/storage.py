"""
Storage utilities for endpoint memory profiling.

Provides functions for:
- Buffering memory profile data in memory
- Flushing buffers to disk efficiently  
- File management (rotation, cleanup)
- Loading and analyzing stored profiles
"""

import asyncio
import json
import os
import threading
from typing import Any, Dict, List, Optional

from litellm._logging import verbose_proxy_logger

from .constants import (
    DEFAULT_FILE_FORMAT,
    DEFAULT_MAX_PROFILES_PER_ENDPOINT,
    DEFAULT_OUTPUT_DIR,
    MEMORY_DECIMAL_PLACES,
)
from .utils import extract_basic_stats, sanitize_endpoint_name


class ProfileBuffer:
    """
    Thread-safe buffer for collecting profile data before flushing to disk.
    
    This class provides efficient buffering to avoid writing to disk on every request,
    which would be prohibitively slow for high-traffic endpoints.
    """
    
    def __init__(
        self,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        max_profiles_per_endpoint: int = DEFAULT_MAX_PROFILES_PER_ENDPOINT,
        file_format: str = DEFAULT_FILE_FORMAT,
    ):
        """
        Initialize profile buffer.
        
        Args:
            output_dir: Directory to store profile files
            max_profiles_per_endpoint: Maximum profiles to keep per endpoint
            file_format: File format ("json" only)
        """
        self.output_dir = output_dir
        self.max_profiles_per_endpoint = max_profiles_per_endpoint
        self.file_format = file_format
        
        # Buffer: {endpoint_name: [profile1, profile2, ...]}
        self._buffer: Dict[str, List[Dict[str, Any]]] = {}
        self._buffer_lock = threading.Lock()
        
        # Track last flush time for each endpoint
        self._last_flush_time: Dict[str, float] = {}
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def add_profile(self, endpoint: str, profile: Dict[str, Any]) -> None:
        """
        Add a profile to the buffer.
        
        Thread-safe method to add profile data without blocking.
        
        Args:
            endpoint: Endpoint path
            profile: Profile dictionary
        """
        with self._buffer_lock:
            if endpoint not in self._buffer:
                self._buffer[endpoint] = []
            self._buffer[endpoint].append(profile)
            
            verbose_proxy_logger.debug(
                f"Added profile to buffer for {endpoint}. "
                f"Buffer size: {len(self._buffer[endpoint])}"
            )
    
    def get_buffer_size(self, endpoint: Optional[str] = None) -> int:
        """
        Get current buffer size.
        
        Args:
            endpoint: Specific endpoint, or None for total across all endpoints
            
        Returns:
            Number of buffered profiles
        """
        with self._buffer_lock:
            if endpoint:
                return len(self._buffer.get(endpoint, []))
            else:
                return sum(len(profiles) for profiles in self._buffer.values())
    
    def get_buffered_endpoints(self) -> List[str]:
        """
        Get list of endpoints with buffered data.
        
        Returns:
            List of endpoint paths
        """
        with self._buffer_lock:
            return list(self._buffer.keys())
    
    def flush_endpoint(self, endpoint: str) -> int:
        """
        Flush buffered profiles for a specific endpoint to disk.
        
        This method:
        1. Extracts buffered data for the endpoint
        2. Loads existing data from disk
        3. Merges and rotates if necessary
        4. Saves to disk using efficient format
        
        Args:
            endpoint: Endpoint path to flush
            
        Returns:
            Number of profiles flushed
        """
        import time
        
        # Get buffered data
        with self._buffer_lock:
            if endpoint not in self._buffer or not self._buffer[endpoint]:
                return 0
            
            buffered_data = self._buffer[endpoint]
            self._buffer[endpoint] = []
        
        if not buffered_data:
            return 0
        
        try:
            # Save to file
            count = self._save_profiles_to_file(endpoint, buffered_data)
            
            # Update flush time
            self._last_flush_time[endpoint] = time.time()
            
            verbose_proxy_logger.info(
                f"Flushed {count} profiles for endpoint {endpoint} to disk"
            )
            
            return count
            
        except Exception as e:
            verbose_proxy_logger.error(
                f"Error flushing profiles for {endpoint}: {e}"
            )
            # Put data back in buffer on error
            with self._buffer_lock:
                self._buffer[endpoint].extend(buffered_data)
            return 0
    
    async def flush_endpoint_async(self, endpoint: str) -> int:
        """
        Async version of flush_endpoint.
        
        Runs flush operation in thread pool to avoid blocking event loop.
        
        Args:
            endpoint: Endpoint path to flush
            
        Returns:
            Number of profiles flushed
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.flush_endpoint, endpoint)
    
    def flush_all(self) -> Dict[str, int]:
        """
        Flush all buffered profiles to disk.
        
        Returns:
            Dictionary mapping endpoint to number of profiles flushed
        """
        endpoints = self.get_buffered_endpoints()
        results = {}
        
        for endpoint in endpoints:
            count = self.flush_endpoint(endpoint)
            if count > 0:
                results[endpoint] = count
        
        return results
    
    async def flush_all_async(self) -> Dict[str, int]:
        """
        Async version of flush_all.
        
        Returns:
            Dictionary mapping endpoint to number of profiles flushed
        """
        endpoints = self.get_buffered_endpoints()
        results = {}
        
        # Flush all endpoints concurrently
        tasks = [self.flush_endpoint_async(ep) for ep in endpoints]
        counts = await asyncio.gather(*tasks, return_exceptions=True)
        
        for endpoint, count in zip(endpoints, counts):
            if isinstance(count, Exception):
                verbose_proxy_logger.error(
                    f"Error flushing {endpoint}: {count}"
                )
            elif isinstance(count, int) and count > 0:
                results[endpoint] = count
        
        return results
    
    def _save_profiles_to_file(
        self,
        endpoint: str,
        buffered_data: List[Dict[str, Any]]
    ) -> int:
        """
        Save buffered profiles to a file.
        
        Args:
            endpoint: Endpoint path
            buffered_data: List of profile dictionaries
            
        Returns:
            Number of profiles saved
        """
        if not buffered_data:
            return 0
        
        # Get safe filename
        safe_filename = sanitize_endpoint_name(endpoint)
        output_file = os.path.join(
            self.output_dir,
            f"{safe_filename}.{self.file_format}"
        )
        
        # Load existing data
        existing_data = self._load_profiles_from_file(output_file)
        
        # Merge with buffered data
        existing_data.extend(buffered_data)
        
        # Rotate if exceeds max
        if self.max_profiles_per_endpoint > 0 and len(existing_data) > self.max_profiles_per_endpoint:
            profiles_to_remove = len(existing_data) - self.max_profiles_per_endpoint
            existing_data = existing_data[-self.max_profiles_per_endpoint:]
            verbose_proxy_logger.debug(
                f"Rotated out {profiles_to_remove} old profiles for {endpoint}"
            )
        
        # Save to file (JSON format only)
        try:
            # Use compact format (no indentation) to reduce file size
            with open(output_file, 'w') as f:
                json.dump(existing_data, f, separators=(',', ':'))
            
            return len(buffered_data)
            
        except Exception as e:
            verbose_proxy_logger.error(
                f"Error saving profiles to {output_file}: {e}"
            )
            raise
    
    def _load_profiles_from_file(self, output_file: str) -> List[Dict[str, Any]]:
        """
        Load existing profiles from a file.
        
        Args:
            output_file: Path to profile file
            
        Returns:
            List of profile dictionaries (empty if file doesn't exist)
        """
        if not os.path.exists(output_file):
            return []
        
        try:
            with open(output_file, 'r') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                verbose_proxy_logger.warning(
                    f"Unexpected format in {output_file}, creating new file"
                )
                return []
            
            return data
            
        except Exception as e:
            verbose_proxy_logger.warning(
                f"Could not load {output_file}: {e}. Creating new file."
            )
            return []
    
    def get_profile_stats(self, endpoint: str) -> Optional[Dict[str, Any]]:
        """
        Get statistics for profiles of a specific endpoint.
        
        Loads from disk and calculates aggregate statistics.
        
        Args:
            endpoint: Endpoint path
            
        Returns:
            Dictionary with statistics, or None if no data
        """
        safe_filename = sanitize_endpoint_name(endpoint)
        output_file = os.path.join(
            self.output_dir,
            f"{safe_filename}.{self.file_format}"
        )
        
        profiles = self._load_profiles_from_file(output_file)
        
        if not profiles:
            return None
        
        stats = extract_basic_stats(profiles)
        
        # Add file info
        try:
            file_size = os.path.getsize(output_file)
            stats['file_size_mb'] = round(
                file_size / (1024 * 1024),
                MEMORY_DECIMAL_PLACES
            )
        except Exception:
            pass
        
        stats['endpoint'] = endpoint
        stats['profile_count'] = len(profiles)
        
        return stats
    
    def cleanup_old_files(self, max_age_seconds: Optional[int] = None) -> int:
        """
        Delete old profile files.
        
        Args:
            max_age_seconds: Delete files older than this (None = delete all)
            
        Returns:
            Number of files deleted
        """
        import time
        
        if not os.path.exists(self.output_dir):
            return 0
        
        deleted_count = 0
        current_time = time.time()
        
        for filename in os.listdir(self.output_dir):
            if not filename.endswith(f".{self.file_format}"):
                continue
            
            filepath = os.path.join(self.output_dir, filename)
            
            try:
                if max_age_seconds is None:
                    # Delete all
                    os.remove(filepath)
                    deleted_count += 1
                else:
                    # Check age
                    file_mtime = os.path.getmtime(filepath)
                    age = current_time - file_mtime
                    if age > max_age_seconds:
                        os.remove(filepath)
                        deleted_count += 1
            except Exception as e:
                verbose_proxy_logger.error(
                    f"Error deleting {filepath}: {e}"
                )
        
        if deleted_count > 0:
            verbose_proxy_logger.info(
                f"Cleaned up {deleted_count} old profile files"
            )
        
        return deleted_count
    
    def list_profile_files(self) -> List[Dict[str, Any]]:
        """
        List all profile files with metadata.
        
        Returns:
            List of dictionaries with file information
        """
        if not os.path.exists(self.output_dir):
            return []
        
        files = []
        
        for filename in os.listdir(self.output_dir):
            if not filename.endswith(f".{self.file_format}"):
                continue
            
            filepath = os.path.join(self.output_dir, filename)
            
            try:
                file_size = os.path.getsize(filepath)
                file_mtime = os.path.getmtime(filepath)
                
                # Load to get count
                profiles = self._load_profiles_from_file(filepath)
                
                files.append({
                    'filename': filename,
                    'endpoint': filename.replace(f".{self.file_format}", ""),
                    'file_size_mb': round(
                        file_size / (1024 * 1024),
                        MEMORY_DECIMAL_PLACES
                    ),
                    'profile_count': len(profiles),
                    'last_modified': file_mtime,
                })
            except Exception as e:
                verbose_proxy_logger.error(
                    f"Error reading {filepath}: {e}"
                )
        
        return sorted(files, key=lambda x: x['last_modified'], reverse=True)


def print_profile_summary(endpoint: str, buffer: ProfileBuffer) -> None:
    """
    Print summary of profiles for an endpoint.
    
    Args:
        endpoint: Endpoint path
        buffer: ProfileBuffer instance
    """
    stats = buffer.get_profile_stats(endpoint)
    
    if not stats:
        print(f"\n[INFO] No profile data found for endpoint: {endpoint}")
        return
    
    print(f"\n{'='*80}")
    print(f"PROFILE SUMMARY: {endpoint}")
    print(f"{'='*80}")
    print(f"Total Requests: {stats['total_requests']}")
    print(f"Average Latency: {stats['avg_latency']:.6f}s")
    print(f"Max Latency: {stats['max_latency']:.6f}s")
    print(f"Min Latency: {stats['min_latency']:.6f}s")
    print(f"Average Memory: {stats['avg_memory']:.3f} MB")
    print(f"Error Rate: {stats['error_rate']:.2f}%")
    print(f"File Size: {stats.get('file_size_mb', 0):.3f} MB")
    print(f"{'='*80}\n")

