"""
Monitoring service for Lago billing integration.

This module provides optional monitoring and health check capabilities
for the Lago billing integration, following the simplified approach
from the technical design document.
"""

import time
from typing import Dict, Any
from .lago_logger import LagoLogger


class MonitoringService:
    """Optional basic monitoring for Lago integration"""
    
    def __init__(self, logger: LagoLogger):
        """
        Initialize monitoring service.
        
        Args:
            logger: LagoLogger instance to monitor
        """
        self.logger = logger
        self.start_time = time.time()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get basic statistics about the Lago integration.
        
        Returns:
            Dictionary with monitoring statistics
        """
        uptime = time.time() - self.start_time
        storage_stats = self.logger.call_store.get_stats()
        
        return {
            "uptime_seconds": int(uptime),
            "stored_calls": storage_stats["total_entries"],
            "config_valid": self.logger.config.is_valid(),
            "lago_api_base": self.logger.config.api_base,
            "publisher_id": self.logger.config.publisher_id,
            "fallback_allow": self.logger.config.fallback_allow,
            "timeout_seconds": self.logger.config.timeout,
            "timestamp": int(time.time())
        }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform basic health check of the Lago integration.
        
        Returns:
            Dictionary with health check results
        """
        config_valid = self.logger.config.is_valid()
        storage_stats = self.logger.call_store.get_stats()
        
        # Determine overall health status
        status = "healthy"
        issues = []
        
        if not config_valid:
            status = "unhealthy"
            issues.append("Invalid configuration")
        
        if storage_stats["total_entries"] > 5000:
            status = "warning"
            issues.append("High number of stored calls")
        
        return {
            "status": status,
            "issues": issues,
            "checks": {
                "lago_configured": bool(self.logger.config.api_base and self.logger.config.api_key),
                "publisher_configured": bool(self.logger.config.publisher_id),
                "storage_operational": True,  # If we can get stats, storage is working
                "config_valid": config_valid
            },
            "metrics": {
                "stored_calls": storage_stats["total_entries"],
                "uptime_seconds": int(time.time() - self.start_time)
            },
            "timestamp": int(time.time())
        }
    
    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get configuration summary (without sensitive data).
        
        Returns:
            Dictionary with configuration information
        """
        return {
            "api_base": self.logger.config.api_base,
            "publisher_id": self.logger.config.publisher_id,
            "timeout": self.logger.config.timeout,
            "fallback_allow": self.logger.config.fallback_allow,
            "api_key_configured": bool(self.logger.config.api_key),
            "config_valid": self.logger.config.is_valid()
        }
