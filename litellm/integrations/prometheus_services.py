# used for monitoring litellm services health on `/metrics` endpoint on LiteLLM Proxy
#### What this does ####
#    On success + failure, log events to Prometheus for litellm / adjacent services (litellm, redis, postgres, llm api providers)


import dotenv, os
import requests  # type: ignore
import traceback
import datetime, subprocess, sys
import litellm, uuid
from litellm._logging import print_verbose, verbose_logger
from litellm.types.services import ServiceLoggerPayload, ServiceTypes


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
                from prometheus_client import Counter, Histogram, REGISTRY
            except ImportError:
                raise Exception(
                    "Missing prometheus_client. Run `pip install prometheus-client`"
                )

            self.Histogram = Histogram
            self.Counter = Counter
            self.REGISTRY = REGISTRY

            verbose_logger.debug(f"in init prometheus services metrics")

            self.services = [item.value for item in ServiceTypes]

            self.payload_to_prometheus_map = (
                {}
            )  # store the prometheus histogram/counter we need to call for each field in payload

            for service in self.services:
                histogram = self.create_histogram(service, type_of_request="latency")
                counter_failed_request = self.create_counter(
                    service, type_of_request="failed_requests"
                )
                counter_total_requests = self.create_counter(
                    service, type_of_request="total_requests"
                )
                self.payload_to_prometheus_map[service] = [
                    histogram,
                    counter_failed_request,
                    counter_total_requests,
                ]

            self.prometheus_to_amount_map: dict = (
                {}
            )  # the field / value in ServiceLoggerPayload the object needs to be incremented by

            ### MOCK TESTING ###
            self.mock_testing = mock_testing
            self.mock_testing_success_calls = 0
            self.mock_testing_failure_calls = 0

        except Exception as e:
            print_verbose(f"Got exception on init prometheus client {str(e)}")
            raise e

    def is_metric_registered(self, metric_name) -> bool:
        for metric in self.REGISTRY.collect():
            if metric_name == metric.name:
                return True
        return False

    def get_metric(self, metric_name):
        for metric in self.REGISTRY.collect():
            for sample in metric.samples:
                if metric_name == sample.name:
                    return metric
        return None

    def create_histogram(self, service: str, type_of_request: str):
        metric_name = "litellm_{}_{}".format(service, type_of_request)
        is_registered = self.is_metric_registered(metric_name)
        if is_registered:
            return self.get_metric(metric_name)
        return self.Histogram(
            metric_name,
            "Latency for {} service".format(service),
            labelnames=[service],
        )

    def create_counter(self, service: str, type_of_request: str):
        metric_name = "litellm_{}_{}".format(service, type_of_request)
        is_registered = self.is_metric_registered(metric_name)
        if is_registered:
            return self.get_metric(metric_name)
        return self.Counter(
            metric_name,
            "Total {} for {} service".format(type_of_request, service),
            labelnames=[service],
        )

    def observe_histogram(
        self,
        histogram,
        labels: str,
        amount: float,
    ):
        assert isinstance(histogram, self.Histogram)

        histogram.labels(labels).observe(amount)

    def increment_counter(
        self,
        counter,
        labels: str,
        amount: float,
    ):
        assert isinstance(counter, self.Counter)

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

    async def async_service_failure_hook(self, payload: ServiceLoggerPayload):
        if self.mock_testing:
            self.mock_testing_failure_calls += 1

        if payload.service.value in self.payload_to_prometheus_map:
            prom_objects = self.payload_to_prometheus_map[payload.service.value]
            for obj in prom_objects:
                if isinstance(obj, self.Counter):
                    self.increment_counter(
                        counter=obj,
                        labels=payload.service.value,
                        amount=1,  # LOG ERROR COUNT TO PROMETHEUS
                    )
