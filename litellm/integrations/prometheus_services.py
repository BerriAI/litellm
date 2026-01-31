# used for monitoring litellm services health on `/metrics` endpoint on LiteLLM Proxy
#### What this does ####
#    On success + failure, log events to Prometheus for litellm / adjacent services (litellm, redis, postgres, llm api providers)


from typing import Dict, List, Optional, Union

from litellm._logging import print_verbose, verbose_logger
from litellm.types.integrations.prometheus import LATENCY_BUCKETS
from litellm.types.services import (
    DEFAULT_SERVICE_CONFIGS,
    ServiceLoggerPayload,
    ServiceMetrics,
    ServiceTypes,
)

FAILED_REQUESTS_LABELS = ["error_class", "function_name"]


class PrometheusServicesLogger:
    # Class variables or attributes
    litellm_service_latency = None  # Class-level attribute to store the Histogram

    def __init__(
        self,
        mock_testing: bool = False,
        **kwargs,
    ):
        try:
            try:
                from prometheus_client import REGISTRY, Counter, Gauge, Histogram
                from prometheus_client.gc_collector import Collector
            except ImportError:
                raise Exception(
                    "Missing prometheus_client. Run `pip install prometheus-client`"
                )

            self.Histogram = Histogram
            self.Counter = Counter
            self.Gauge = Gauge
            self.REGISTRY = REGISTRY

            verbose_logger.debug("in init prometheus services metrics")

            self.payload_to_prometheus_map: Dict[
                str, List[Union[Histogram, Counter, Gauge, Collector]]
            ] = {}

            for service in ServiceTypes:
                service_metrics: List[Union[Histogram, Counter, Gauge, Collector]] = []

                metrics_to_initialize = self._get_service_metrics_initialize(service)

                # Initialize only the configured metrics for each service
                if ServiceMetrics.HISTOGRAM in metrics_to_initialize:
                    histogram = self.create_histogram(
                        service.value, type_of_request="latency"
                    )
                    if histogram:
                        service_metrics.append(histogram)

                if ServiceMetrics.COUNTER in metrics_to_initialize:
                    counter_failed_request = self.create_counter(
                        service.value,
                        type_of_request="failed_requests",
                        additional_labels=FAILED_REQUESTS_LABELS,
                    )
                    if counter_failed_request:
                        service_metrics.append(counter_failed_request)
                    counter_total_requests = self.create_counter(
                        service.value, type_of_request="total_requests"
                    )
                    if counter_total_requests:
                        service_metrics.append(counter_total_requests)

                if ServiceMetrics.GAUGE in metrics_to_initialize:
                    gauge = self.create_gauge(service.value, type_of_request="size")
                    if gauge:
                        service_metrics.append(gauge)

                if service_metrics:
                    self.payload_to_prometheus_map[service.value] = service_metrics

            self.prometheus_to_amount_map: dict = {}
            ### MOCK TESTING ###
            self.mock_testing = mock_testing
            self.mock_testing_success_calls = 0
            self.mock_testing_failure_calls = 0

        except Exception as e:
            print_verbose(f"Got exception on init prometheus client {str(e)}")
            raise e

    def _get_service_metrics_initialize(
        self, service: ServiceTypes
    ) -> List[ServiceMetrics]:
        DEFAULT_METRICS = [ServiceMetrics.COUNTER, ServiceMetrics.HISTOGRAM]
        if service not in DEFAULT_SERVICE_CONFIGS:
            return DEFAULT_METRICS

        metrics = DEFAULT_SERVICE_CONFIGS.get(service, {}).get("metrics", [])
        if not metrics:
            verbose_logger.debug(f"No metrics found for service {service}")
            return DEFAULT_METRICS
        return metrics

    def is_metric_registered(self, metric_name) -> bool:
        for metric in self.REGISTRY.collect():
            if metric_name == metric.name:
                return True
        return False

    def _get_metric(self, metric_name):
        """
        Helper function to get a metric from the registry by name.
        """
        return self.REGISTRY._names_to_collectors.get(metric_name)

    def create_histogram(self, service: str, type_of_request: str):
        metric_name = "litellm_{}_{}".format(service, type_of_request)
        is_registered = self.is_metric_registered(metric_name)
        if is_registered:
            return self._get_metric(metric_name)
        return self.Histogram(
            metric_name,
            "Latency for {} service".format(service),
            labelnames=[service],
            buckets=LATENCY_BUCKETS,
        )

    def create_gauge(self, service: str, type_of_request: str):
        metric_name = "litellm_{}_{}".format(service, type_of_request)
        is_registered = self.is_metric_registered(metric_name)
        if is_registered:
            return self._get_metric(metric_name)
        return self.Gauge(
            metric_name, "Gauge for {} service".format(service), labelnames=[service]
        )

    def create_counter(
        self,
        service: str,
        type_of_request: str,
        additional_labels: Optional[List[str]] = None,
    ):
        metric_name = "litellm_{}_{}".format(service, type_of_request)
        is_registered = self.is_metric_registered(metric_name)
        if is_registered:
            return self._get_metric(metric_name)
        return self.Counter(
            metric_name,
            "Total {} for {} service".format(type_of_request, service),
            labelnames=[service] + (additional_labels or []),
        )

    def observe_histogram(
        self,
        histogram,
        labels: str,
        amount: float,
    ):
        assert isinstance(histogram, self.Histogram)

        histogram.labels(labels).observe(amount)

    def update_gauge(
        self,
        gauge,
        labels: str,
        amount: float,
    ):
        assert isinstance(gauge, self.Gauge)
        gauge.labels(labels).set(amount)

    def increment_counter(
        self,
        counter,
        labels: str,
        amount: float,
        additional_labels: Optional[List[str]] = [],
    ):
        assert isinstance(counter, self.Counter)

        if additional_labels:
            counter.labels(labels, *additional_labels).inc(amount)
        else:
            counter.labels(labels).inc(amount)

    def service_success_hook(self, payload: ServiceLoggerPayload):
        if self.mock_testing:
            self.mock_testing_success_calls += 1

        if payload.service.value in self.payload_to_prometheus_map:
            prom_objects = self.payload_to_prometheus_map[payload.service.value]
            for obj in prom_objects:
                if isinstance(obj, self.Histogram):
                    self.observe_histogram(
                        histogram=obj,
                        labels=payload.service.value,
                        amount=payload.duration,
                    )
                elif isinstance(obj, self.Counter) and "total_requests" in obj._name:
                    self.increment_counter(
                        counter=obj,
                        labels=payload.service.value,
                        amount=1,  # LOG TOTAL REQUESTS TO PROMETHEUS
                    )

    def service_failure_hook(self, payload: ServiceLoggerPayload):
        if self.mock_testing:
            self.mock_testing_failure_calls += 1

        if payload.service.value in self.payload_to_prometheus_map:
            prom_objects = self.payload_to_prometheus_map[payload.service.value]
            for obj in prom_objects:
                if isinstance(obj, self.Counter):
                    self.increment_counter(
                        counter=obj,
                        labels=payload.service.value,
                        amount=1,  # LOG ERROR COUNT / TOTAL REQUESTS TO PROMETHEUS
                    )

    async def async_service_success_hook(self, payload: ServiceLoggerPayload):
        """
        Log successful call to prometheus
        """
        if self.mock_testing:
            self.mock_testing_success_calls += 1

        if payload.service.value in self.payload_to_prometheus_map:
            prom_objects = self.payload_to_prometheus_map[payload.service.value]
            for obj in prom_objects:
                if isinstance(obj, self.Histogram):
                    self.observe_histogram(
                        histogram=obj,
                        labels=payload.service.value,
                        amount=payload.duration,
                    )
                elif isinstance(obj, self.Counter) and "total_requests" in obj._name:
                    self.increment_counter(
                        counter=obj,
                        labels=payload.service.value,
                        amount=1,  # LOG TOTAL REQUESTS TO PROMETHEUS
                    )
                elif isinstance(obj, self.Gauge):
                    if payload.event_metadata:
                        self.update_gauge(
                            gauge=obj,
                            labels=payload.event_metadata.get("gauge_labels") or "",
                            amount=payload.event_metadata.get("gauge_value") or 0,
                        )

    async def async_service_failure_hook(
        self,
        payload: ServiceLoggerPayload,
        error: Union[str, Exception],
    ):
        if self.mock_testing:
            self.mock_testing_failure_calls += 1
        error_class = error.__class__.__name__
        function_name = payload.call_type

        if payload.service.value in self.payload_to_prometheus_map:
            prom_objects = self.payload_to_prometheus_map[payload.service.value]
            for obj in prom_objects:
                # increment both failed and total requests
                if isinstance(obj, self.Counter):
                    if "failed_requests" in obj._name:
                        self.increment_counter(
                            counter=obj,
                            labels=payload.service.value,
                            # log additional_labels=["error_class", "function_name"], used for debugging what's going wrong with the DB
                            additional_labels=[error_class, function_name],
                            amount=1,  # LOG ERROR COUNT TO PROMETHEUS
                        )
                    else:
                        self.increment_counter(
                            counter=obj,
                            labels=payload.service.value,
                            amount=1,  # LOG TOTAL REQUESTS TO PROMETHEUS
                        )
