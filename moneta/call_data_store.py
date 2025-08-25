"""
CallDataStore - Thread-safe storage for call metadata in Lago billing integration.

This module provides minimal thread-safe storage for mapping call IDs to customer IDs
during the request lifecycle, enabling data flow between pre-call entitlement checks
and post-call usage reporting.
"""

import threading
import time
from typing import Dict, Optional, Tuple


class CallDataStore:
    """Minimal thread-safe storage for call data"""
    
    def __init__(self, max_age_seconds: int = 3600):
        """
        Initialize the call data store.
        
        Args:
            max_age_seconds: Maximum age for stored entries before cleanup (default: 1 hour)
        """
        self._data: Dict[str, Tuple[str, float]] = {}  # call_id -> (customer_id, timestamp)
        self._lock = threading.Lock()
        self._max_age = max_age_seconds
    
    def store(self, call_id: str, customer_id: str) -> None:
        """
        Store customer ID for a call.
        
        Args:
            call_id: Unique identifier for the LLM API call
            customer_id: Customer identifier for billing
        """
        with self._lock:
            self._data[call_id] = (customer_id, time.time())
            self._cleanup_if_needed()
    
    def get_and_remove(self, call_id: str) -> Optional[str]:
        """
        Get and remove customer ID for a call.
        
        Args:
            call_id: Unique identifier for the LLM API call
            
        Returns:
            Customer ID if found, None otherwise
        """
        with self._lock:
            if call_id in self._data:
                customer_id, _ = self._data.pop(call_id)
                return customer_id
        return None
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get storage statistics for monitoring.
        
        Returns:
            Dictionary with storage statistics
        """
        with self._lock:
            return {
                "total_entries": len(self._data),
                "max_age_seconds": self._max_age
            }
    
    def _cleanup_if_needed(self) -> None:
        """
        Simple cleanup - remove entries older than max_age.
        Runs every 100 entries to balance performance and memory usage.
        """
        if len(self._data) % 100 == 0:  # Cleanup every 100 entries
            current_time = time.time()
            expired = [
                call_id for call_id, (_, timestamp) in self._data.items()
                if current_time - timestamp > self._max_age
            ]
            for call_id in expired:
                self._data.pop(call_id, None)
            
            if expired:
                print(f"CallDataStore: Cleaned up {len(expired)} expired entries")
