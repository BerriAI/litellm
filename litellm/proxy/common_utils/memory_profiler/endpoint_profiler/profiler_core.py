"""
Core EndpointProfiler singleton class for managing profiling state.

This module provides the central profiler that:
- Manages profiling state (enabled/disabled, sampling rate)
- Maintains request counters
- Coordinates buffer management and flushing
- Provides management API
- Handles tracemalloc lifecycle
"""

import asyncio
import threading
import tracemalloc
from typing import Any, Optional

from litellm._logging import verbose_proxy_logger

from .capture import capture_request_profile
from .constants import (
    DEFAULT_ASYNC_FLUSH,
    DEFAULT_BUFFER_SIZE,
    DEFAULT_FLUSH_INTERVAL_SECONDS,
    DEFAULT_PROFILING_ENABLED,
    DEFAULT_SAMPLING_RATE,
)
from .storage import ProfileBuffer
from .utils import should_sample_request


class EndpointProfiler:
    """
    Manages endpoint memory profiling for the proxy server.
    
    This class:
    - Maintains profiling state (enabled/disabled, sampling rate)
    - Buffers profile data in memory
    - Periodically flushes to disk
    - Provides management API
    - Handles tracemalloc lifecycle
    """
    
    _instance: Optional['EndpointProfiler'] = None
    _instance_lock = threading.Lock()
    
    def __init__(
        self,
        enabled: bool = DEFAULT_PROFILING_ENABLED,
        sampling_rate: float = DEFAULT_SAMPLING_RATE,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        flush_interval: int = DEFAULT_FLUSH_INTERVAL_SECONDS,
        async_flush: bool = DEFAULT_ASYNC_FLUSH,
        **kwargs
    ):
        """
        Initialize endpoint memory profiler.
        
        Args:
            enabled: Whether profiling is enabled
            sampling_rate: Sampling rate (0.0 to 1.0)
            buffer_size: Number of profiles to buffer before flushing
            flush_interval: Seconds between forced flushes
            async_flush: Whether to flush asynchronously
            **kwargs: Additional arguments passed to ProfileBuffer
        """
        self.enabled = enabled
        self.sampling_rate = sampling_rate
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.async_flush = async_flush
        
        # Request counter
        self._request_counter = 0
        self._counter_lock = threading.Lock()
        
        # Track last detailed capture memory for smart capture
        self._last_detailed_memory: dict = {}
        self._last_detailed_lock = threading.Lock()
        
        # Buffer
        self.buffer = ProfileBuffer(**kwargs)
        
        # Flush task
        self._flush_task: Optional[asyncio.Task] = None
        self._flush_task_lock = threading.Lock()
        self._stop_flush_task = False
        
        # Start tracemalloc for memory profiling
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            verbose_proxy_logger.info("Started tracemalloc for endpoint memory profiling")
        
        verbose_proxy_logger.info(
            f"EndpointProfiler initialized: enabled={enabled}, "
            f"sampling_rate={sampling_rate}, buffer_size={buffer_size}"
        )
    
    @classmethod
    def get_instance(cls, **kwargs) -> 'EndpointProfiler':
        """
        Get or create singleton instance of EndpointProfiler.
        
        Args:
            **kwargs: Arguments passed to constructor on first call
            
        Returns:
            EndpointProfiler instance
        """
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls(**kwargs)
        return cls._instance
    
    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (useful for testing)."""
        with cls._instance_lock:
            if cls._instance:
                cls._instance.stop()
            cls._instance = None
    
    def start(self) -> None:
        """Enable profiling."""
        self.enabled = True
        verbose_proxy_logger.info("Endpoint profiling enabled")
    
    def stop(self) -> None:
        """Disable profiling and flush all buffers."""
        self.enabled = False
        self._stop_flush_task = True
        
        # Flush all buffered data
        self.buffer.flush_all()
        
        verbose_proxy_logger.info("Endpoint profiling disabled")
    
    def set_sampling_rate(self, rate: float) -> None:
        """
        Update sampling rate.
        
        Args:
            rate: New sampling rate (0.0 to 1.0)
        """
        if not 0.0 <= rate <= 1.0:
            raise ValueError("Sampling rate must be between 0.0 and 1.0")
        
        self.sampling_rate = rate
        verbose_proxy_logger.info(f"Sampling rate updated to {rate}")
    
    def increment_counter(self) -> int:
        """
        Increment and return request counter.
        
        Returns:
            New counter value
        """
        with self._counter_lock:
            self._request_counter += 1
            return self._request_counter
    
    def should_profile_request(self, request_counter: int) -> bool:
        """
        Determine if request should be profiled based on settings.
        
        Args:
            request_counter: Current request counter
            
        Returns:
            True if request should be profiled
        """
        if not self.enabled:
            return False
        
        return should_sample_request(request_counter, self.sampling_rate)
    
    def get_last_detailed_memory(self, endpoint: str) -> float:
        """
        Get memory at last detailed capture for an endpoint.
        
        Args:
            endpoint: Endpoint path
            
        Returns:
            Memory in MB (0.0 if never captured)
        """
        with self._last_detailed_lock:
            return self._last_detailed_memory.get(endpoint, 0.0)
    
    def set_last_detailed_memory(self, endpoint: str, memory_mb: float) -> None:
        """
        Update memory at last detailed capture for an endpoint.
        
        Args:
            endpoint: Endpoint path
            memory_mb: Memory in MB
        """
        with self._last_detailed_lock:
            self._last_detailed_memory[endpoint] = memory_mb
    
    def add_profile(
        self,
        endpoint: str,
        method: str,
        request_counter: int,
        start_time: float,
        end_time: float,
        status_code: int,
        error_message: Optional[str] = None,
        request: Optional[Any] = None,
        response: Optional[Any] = None,
    ) -> None:
        """
        Add a profile to the buffer.
        
        Args:
            endpoint: Endpoint path
            method: HTTP method
            request_counter: Request counter
            start_time: Request start time
            end_time: Request end time
            status_code: HTTP status code
            error_message: Error message if any
            request: FastAPI Request object
            response: Response object
        """
        # Capture profile
        last_detailed_memory = self.get_last_detailed_memory(endpoint)
        
        profile = capture_request_profile(
            endpoint=endpoint,
            method=method,
            request_counter=request_counter,
            start_time=start_time,
            end_time=end_time,
            status_code=status_code,
            error_message=error_message,
            _request=request,
            _response=response,
            last_detailed_memory_mb=last_detailed_memory,
            track_memory=True,  # Always track memory
        )
        
        # Update last detailed memory if we captured a detailed snapshot
        if profile.get('detailed_snapshot') and 'memory' in profile:
            memory_mb = profile['memory'].get('current_mb', 0.0)
            self.set_last_detailed_memory(endpoint, memory_mb)
        
        # Add to buffer
        self.buffer.add_profile(endpoint, profile)
        
        # Optimize: Check buffer size without lock first for fast path
        # Only flush if we hit the threshold (avoid unnecessary lock acquisition)
        # get_buffer_size already handles locking internally
        buffer_size = self.buffer.get_buffer_size(endpoint)
        if buffer_size >= self.buffer_size:
            verbose_proxy_logger.debug(
                f"Buffer size ({buffer_size}) reached threshold ({self.buffer_size}), "
                f"flushing {endpoint}"
            )
            if self.async_flush:
                # Schedule async flush (don't await, just fire and forget)
                asyncio.create_task(self.buffer.flush_endpoint_async(endpoint))
            else:
                # Flush synchronously
                self.buffer.flush_endpoint(endpoint)
    
    async def start_auto_flush(self) -> None:
        """
        Start automatic flushing task.
        
        This task periodically flushes buffers based on flush_interval.
        Should be called once when the server starts.
        """
        if self.flush_interval <= 0:
            verbose_proxy_logger.info("Auto-flush disabled (flush_interval <= 0)")
            return
        
        with self._flush_task_lock:
            if self._flush_task is not None:
                verbose_proxy_logger.warning("Auto-flush task already running")
                return
            
            self._flush_task = asyncio.create_task(self._auto_flush_loop())
        
        verbose_proxy_logger.info(
            f"Started auto-flush task (interval: {self.flush_interval}s)"
        )
    
    async def stop_auto_flush(self) -> None:
        """Stop automatic flushing task."""
        self._stop_flush_task = True
        
        with self._flush_task_lock:
            if self._flush_task:
                self._flush_task.cancel()
                try:
                    await self._flush_task
                except asyncio.CancelledError:
                    pass
                self._flush_task = None
        
        verbose_proxy_logger.info("Stopped auto-flush task")
    
    async def _auto_flush_loop(self) -> None:
        """Background task that periodically flushes buffers."""
        while not self._stop_flush_task:
            try:
                await asyncio.sleep(self.flush_interval)
                
                if self._stop_flush_task:
                    break
                
                # Flush all endpoints
                results = await self.buffer.flush_all_async()
                
                if results:
                    total_flushed = sum(results.values())
                    verbose_proxy_logger.debug(
                        f"Auto-flush: flushed {total_flushed} profiles "
                        f"across {len(results)} endpoints"
                    )
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                verbose_proxy_logger.error(f"Error in auto-flush loop: {e}")
                # Continue loop even on error
    
    def get_stats(self) -> dict:
        """
        Get current profiler statistics.
        
        Returns:
            Dictionary with profiler stats
        """
        return {
            'enabled': self.enabled,
            'sampling_rate': self.sampling_rate,
            'total_requests': self._request_counter,
            'buffer_size': self.buffer.get_buffer_size(),
            'buffered_endpoints': len(self.buffer.get_buffered_endpoints()),
            'tracemalloc_enabled': tracemalloc.is_tracing(),
        }

