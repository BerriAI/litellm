# used for monitoring litellm services health on `/metrics` endpoint on LiteLLM Proxy
#### What this does ####
#    On success + failure, log events to Prometheus for litellm / adjacent services (litellm, redis, postgres, llm api providers)


import dotenv, os
import requests

dotenv.load_dotenv()  # Loading env variables using dotenv
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
            from prometheus_client import Counter, Histogram, REGISTRY

            self.Histogram = Histogram
            self.Counter = Counter
            self.REGISTRY = REGISTRY

            verbose_logger.debug(f"in init prometheus services metrics")

            self.services = [item.value for item in ServiceTypes]

            self.payload_to_prometheus_map = (
                {}
            )  # store the prometheus histogram/counter we need to call for each field in payload

            for service in self.services:
                histogram = self.create_histogram(service)
                counter = self.create_counter(service)
                self.payload_to_prometheus_map[service] = [histogram, counter]

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
            print(f"metric name: {metric.name}")
            if metric_name == metric.name:
                return True
        return False

    def get_metric(self, metric_name):
        for metric in self.REGISTRY.collect():
            for sample in metric.samples:
                if metric_name == sample.name:
                    return metric
        return None

    def create_histogram(self, label: str):
        metric_name = "litellm_{}_latency".format(label)
        is_registered = self.is_metric_registered(metric_name)
        if is_registered:
            return self.get_metric(metric_name)
        return self.Histogram(
            metric_name,
            "Latency for {} service".format(label),
            labelnames=[label],
        )

    def create_counter(self, label: str):
        metric_name = "litellm_{}_requests".format(label)
        is_registered = self.is_metric_registered(metric_name)
        if is_registered:
            return self.get_metric(metric_name)
        return self.Counter(
            metric_name,
            "Total failed requests for {} service".format(label),
            labelnames=[label],
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
                        amount=1,  # LOG ERROR COUNT TO PROMETHEUS
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
