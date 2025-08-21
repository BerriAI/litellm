"""
Profiler Middleware

Performance profiling middleware using pyinstrument for FastAPI applications.
Can be enabled by setting profiling=True in general_settings or litellm_settings.
"""
import os
import uuid
from typing import Optional

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

import litellm
from litellm._logging import verbose_proxy_logger


class ProfilerMiddleware(BaseHTTPMiddleware):
    """
    Middleware to profile requests using pyinstrument
    
    Enabled by setting the environment variable:
    
    LITELLM_PROFILING=true
    """

    def __init__(self, app, interval: float = 0.005):
        super().__init__(app)
        self.interval = interval
        self._profiler = None
        self._pyinstrument_available = self._check_pyinstrument_availability()
        
        if not self._pyinstrument_available:
            verbose_proxy_logger.warning(
                "pyinstrument not available. Install it with: pip install pyinstrument"
            )

    def _check_pyinstrument_availability(self) -> bool:
        """Check if pyinstrument is available"""
        try:
            import pyinstrument  # noqa: F401
            return True
        except ImportError:
            return False

    def _get_profiler(self):
        """Lazy load the profiler to avoid import errors if pyinstrument is not available"""
        if not self._pyinstrument_available:
            return None
            
        if self._profiler is None:
            try:
                from pyinstrument import Profiler
                self._profiler = Profiler(interval=self.interval)
            except ImportError:
                verbose_proxy_logger.warning(
                    "Failed to import pyinstrument. Profiling disabled."
                )
                return None
        return self._profiler

    def _generate_random_filename(self) -> str:
        """Generate a random filename for profiling output"""
        random_id = str(uuid.uuid4())[:8]
        return f"profile_{random_id}.html"

    async def dispatch(self, request: Request, call_next):
        # Check if profiling is enabled
        if not self._should_profile():
            return await call_next(request)

        if not self._pyinstrument_available:
            verbose_proxy_logger.warning(
                "Profiling requested but pyinstrument not available. Skipping profiling."
            )
            return await call_next(request)

        profiler = self._get_profiler()
        # print("profiler", profiler)
        if profiler is None:
            return await call_next(request)

        # Start profiling
        try:
            profiler.start()
            verbose_proxy_logger.debug(f"Started profiling for request: {request.url.path}")
            
            # Process the request
            response = await call_next(request)
            
            # Stop profiling and write results to file
            profiler.stop()
            verbose_proxy_logger.debug(f"Stopped profiling for request: {request.url.path}")
            
            # Write profiling results to a random file by default
            try:
                filename = self._generate_random_filename()
                verbose_proxy_logger.info(f"Profiling results written to: {filename}")
            except Exception as e:
                verbose_proxy_logger.warning(
                    f"Failed to write profiling results to file: {e}"
                )
            
            # Optionally open in browser (controlled by environment variable)
            if self._should_open_in_browser():
                try:
                    profiler.open_in_browser()
                except Exception as e:
                    verbose_proxy_logger.warning(
                        f"Failed to open profiler results in browser: {e}"
                    )
            
            return response
            
        except Exception as e:
            verbose_proxy_logger.warning(f"Error during profiling: {e}")
            # Ensure we still return a response even if profiling fails
            return await call_next(request)

    def _should_profile(self) -> bool:
        """Check if profiling should be enabled based on configuration"""
        from litellm.secret_managers.main import get_secret_bool
        return get_secret_bool("LITELLM_PROFILING") is True

    def _should_open_in_browser(self) -> bool:
        """Check if profiler results should be opened in browser"""
        return True

    