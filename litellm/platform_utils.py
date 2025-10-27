"""
Platform detection and optimization utilities for LiteLLM.

This module provides platform-specific optimizations, particularly for macOS/Apple Silicon.
All functions maintain backward compatibility with existing behavior.
"""

import os
import platform
import sys
from typing import Optional, Dict, Any


def is_apple_silicon() -> bool:
    """
    Detect if running on Apple Silicon (ARM64) architecture.

    Returns:
        bool: True if running on Apple Silicon, False otherwise.
    """
    return sys.platform == "darwin" and platform.machine() == "arm64"


def is_macos() -> bool:
    """
    Detect if running on macOS (any architecture).

    Returns:
        bool: True if running on macOS, False otherwise.
    """
    return sys.platform == "darwin"


def get_cpu_count() -> int:
    """
    Get the number of CPU cores available.

    Returns:
        int: Number of CPU cores, or 4 as a fallback.
    """
    try:
        return os.cpu_count() or 4
    except Exception:
        return 4


def get_optimal_worker_count() -> int:
    """
    Calculate optimal number of workers based on system architecture.

    For Apple Silicon:
        - Uses P-core count (typically half of total cores)
        - Caps at 8 to avoid oversubscription
        - Minimum of 2 workers

    For other platforms:
        - Uses CPU count with reasonable defaults
        - Caps at 8 for stability

    Returns:
        int: Recommended number of workers for the proxy server.
    """
    cpu_count = get_cpu_count()

    if is_apple_silicon():
        # Apple Silicon has P-cores (performance) and E-cores (efficiency)
        # Most common configs: 8-core (4P+4E), 10-core (4P+6E), etc.
        # Use P-core count for optimal performance
        if cpu_count > 4:
            return min(cpu_count // 2, 8)
        else:
            return 2

    # Standard calculation for other platforms
    if cpu_count >= 8:
        return 8
    elif cpu_count >= 4:
        return 4
    else:
        return 2


def get_optimal_thread_pool_workers() -> int:
    """
    Calculate optimal thread pool workers for concurrent operations.

    For Apple Silicon:
        - Uses 1-2x CPU count (uvloop handles I/O efficiently)
        - Caps at 32 threads

    For other platforms:
        - Uses standard CPU count + 4 formula
        - Caps at 32 threads

    Returns:
        int: Recommended number of thread pool workers.
    """
    cpu_count = get_cpu_count()

    if is_apple_silicon():
        # Apple Silicon with uvloop: I/O is handled efficiently by the event loop
        # Conservative threading to avoid context switching overhead
        return min(cpu_count * 2, 32)

    # Standard calculation: CPU count + 4 (common for I/O-bound workloads)
    return min(cpu_count + 4, 32)


def get_optimal_connection_pool_limit() -> int:
    """
    Calculate optimal HTTP connection pool limit.

    For Apple Silicon:
        - Uses more aggressive pooling (256 connections)
        - Leverages efficient memory and network handling

    For other platforms:
        - Uses unlimited (0) as default for maximum flexibility

    Returns:
        int: Recommended connection pool limit (0 = unlimited).
    """
    if is_apple_silicon():
        # Apple Silicon: More aggressive connection pooling
        # 256 is a good balance between performance and resource usage
        return 256

    # Default: unlimited connections (existing behavior)
    return 0


def get_optimal_keepalive_timeout() -> int:
    """
    Calculate optimal HTTP keepalive timeout in seconds.

    For Apple Silicon:
        - Uses shorter timeout (60s) for better resource utilization
        - Apple Silicon handles reconnections efficiently

    For other platforms:
        - Uses standard 120s timeout

    Returns:
        int: Recommended keepalive timeout in seconds.
    """
    if is_apple_silicon():
        # Apple Silicon: Shorter keepalive for better resource management
        return 60

    # Standard keepalive timeout
    return 120


def get_optimal_dns_cache_ttl() -> int:
    """
    Calculate optimal DNS cache TTL in seconds.

    For Apple Silicon:
        - Uses longer cache (600s = 10min) on stable networks
        - Reduces DNS lookup overhead

    For other platforms:
        - Uses standard 300s (5min) timeout

    Returns:
        int: Recommended DNS cache TTL in seconds.
    """
    if is_apple_silicon():
        # Apple Silicon: Longer DNS cache on typically stable networks
        return 600

    # Standard DNS cache TTL
    return 300


def get_optimal_db_pool_size() -> int:
    """
    Calculate optimal database connection pool size.

    For Apple Silicon:
        - Uses 15 connections (optimal for most workloads)

    For other platforms:
        - Uses 10 connections as standard

    Returns:
        int: Recommended database connection pool size.
    """
    if is_apple_silicon():
        # Apple Silicon: Slightly larger pool for better concurrency
        return 15

    # Standard pool size
    return 10


def get_ssl_cipher_priority() -> Optional[str]:
    """
    Get SSL/TLS cipher suite priority for the platform.

    For Apple Silicon:
        - Prioritizes ChaCha20-Poly1305 (hardware-accelerated on ARM)
        - Falls back to AES-GCM variants

    For other platforms:
        - Returns None to use default cipher configuration

    Returns:
        Optional[str]: Custom cipher string for Apple Silicon, None for default.
    """
    if is_apple_silicon():
        # Apple Silicon: Prioritize ChaCha20 (hardware-accelerated)
        return (
            "TLS_CHACHA20_POLY1305_SHA256:"
            "TLS_AES_256_GCM_SHA384:"
            "TLS_AES_128_GCM_SHA256:"
            "ECDHE-RSA-CHACHA20-POLY1305:"
            "ECDHE-ECDSA-CHACHA20-POLY1305:"
            "ECDHE-RSA-AES256-GCM-SHA384:"
            "ECDHE-RSA-AES128-GCM-SHA256:"
            "ECDHE-ECDSA-AES256-GCM-SHA384:"
            "ECDHE-ECDSA-AES128-GCM-SHA256"
        )

    # Use default ciphers for other platforms
    return None


def get_platform_info() -> Dict[str, Any]:
    """
    Get comprehensive platform information for logging and debugging.

    Returns:
        Dict[str, Any]: Dictionary containing platform details.
    """
    return {
        "platform": sys.platform,
        "architecture": platform.machine(),
        "cpu_count": get_cpu_count(),
        "python_version": platform.python_version(),
        "is_apple_silicon": is_apple_silicon(),
        "is_macos": is_macos(),
        "optimal_workers": get_optimal_worker_count(),
        "optimal_thread_pool": get_optimal_thread_pool_workers(),
    }


def log_platform_optimization_info(logger=None) -> None:
    """
    Log platform-specific optimization information.

    Args:
        logger: Optional logger instance. If None, uses print.
    """
    info = get_platform_info()

    log_func = logger.info if logger else print

    log_func(f"Platform: {info['platform']} ({info['architecture']})")
    log_func(f"CPU Count: {info['cpu_count']}")

    if info["is_apple_silicon"]:
        log_func("✓ Apple Silicon detected - optimizations enabled")
        log_func(f"  - Recommended workers: {info['optimal_workers']}")
        log_func(f"  - Thread pool size: {info['optimal_thread_pool']}")
        log_func(f"  - Connection pool: {get_optimal_connection_pool_limit()}")
        log_func(f"  - Keepalive timeout: {get_optimal_keepalive_timeout()}s")
        log_func(f"  - DNS cache TTL: {get_optimal_dns_cache_ttl()}s")
        log_func("  - SSL: ChaCha20-Poly1305 prioritized (hardware-accelerated)")
    elif info["is_macos"]:
        log_func(f"✓ macOS detected ({info['architecture']})")
        log_func(f"  - Recommended workers: {info['optimal_workers']}")
    else:
        log_func("Using standard configuration")


# Environment variable helpers
def get_env_with_platform_default(
    env_var: str,
    platform_default: int,
    standard_default: int,
) -> int:
    """
    Get environment variable with platform-aware defaults.

    Args:
        env_var: Environment variable name
        platform_default: Default value for Apple Silicon
        standard_default: Default value for other platforms

    Returns:
        int: Environment variable value or platform-appropriate default
    """
    if env_var in os.environ:
        return int(os.getenv(env_var))

    if is_apple_silicon():
        return platform_default

    return standard_default
