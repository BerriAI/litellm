"""
Main endpoint memory profiler module.

This module re-exports the core profiler components for backward compatibility.
The actual implementations are in:
- profiler_core.py: EndpointProfiler singleton class
- decorator.py: profile_endpoint decorator

Provides:
- EndpointProfiler class for managing memory profiling state
- Decorator for profiling endpoint memory usage
- Automatic buffer flushing
- Management API for controlling profiler
"""

# Re-export core components for backward compatibility
from .profiler_core import EndpointProfiler
from .decorator import profile_endpoint

__all__ = ['EndpointProfiler', 'profile_endpoint']
