"""
Prometheus metric helpers for the auth layer.

All metrics are thin wrappers around the shared ``PrometheusLogger`` instance so
that every counter follows the same registration path (``_counter_factory``,
label-filter config) as the rest of LiteLLM's metrics.

Usage::

    from litellm.proxy.auth.auth_metrics import AuthMetrics

    AuthMetrics.inc_combined_view_query(hashed_token="sk-xxx")
"""

from litellm._logging import verbose_proxy_logger
from litellm.types.integrations.prometheus import UserAPIKeyLabelValues


class AuthMetrics:
    """Static helpers for incrementing auth-layer Prometheus counters."""

    @staticmethod
    def _get_prom():
        """Return the active PrometheusLogger, or None if Prometheus is not configured."""
        try:
            from litellm.router_utils.cooldown_callbacks import (
                _get_prometheus_logger_from_callbacks,
            )

            return _get_prometheus_logger_from_callbacks()
        except Exception:
            return None

    @staticmethod
    def inc_combined_view_query(hashed_token: str) -> None:
        """
        Increment ``litellm_auth_combined_view_queries_total``.

        Called once per virtual-key DB lookup (combined_view query). Each
        increment represents a cache miss that hit the database — use this to
        validate that ``enable_redis_auth_cache`` is reducing DB load.
        """
        try:
            prom = AuthMetrics._get_prom()
            if prom is not None:
                # Counter labelnames include ``hashed_api_key`` plus any
                # ``custom_prometheus_metadata_labels`` / ``custom_prometheus_tags``
                # (see ``PrometheusMetricLabels.get_labels``). Use the same
                # ``_inc_labeled_counter`` + ``prometheus_label_factory`` path as
                # other metrics so label cardinality always matches registration.
                prom._inc_labeled_counter(
                    prom.litellm_auth_combined_view_queries_total,
                    "litellm_auth_combined_view_queries_total",
                    UserAPIKeyLabelValues(hashed_api_key=hashed_token),
                )
        except Exception as e:
            verbose_proxy_logger.debug(
                "AuthMetrics.inc_combined_view_query: failed to increment counter: %s",
                e,
            )
