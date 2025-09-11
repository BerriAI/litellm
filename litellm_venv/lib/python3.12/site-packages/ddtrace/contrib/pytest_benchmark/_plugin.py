import pytest

from ddtrace.contrib.pytest._utils import _extract_span
from ddtrace.contrib.pytest_benchmark.constants import BENCHMARK_INFO
from ddtrace.contrib.pytest_benchmark.constants import PLUGIN_METRICS
from ddtrace.contrib.pytest_benchmark.constants import PLUGIN_OUTLIERS
from ddtrace.ext.test import TEST_TYPE


class _PytestBenchmarkPlugin:
    @pytest.hookimpl()
    def pytest_runtest_makereport(self, item, call):
        fixture = hasattr(item, "funcargs") and item.funcargs.get("benchmark")
        if fixture and fixture.stats:
            stat_object = fixture.stats.stats
            span = _extract_span(item)

            if span is None:
                return

            span.set_tag_str(TEST_TYPE, "benchmark")
            span.set_tag_str(BENCHMARK_INFO, "Time")
            for span_path, tag in PLUGIN_METRICS.items():
                if hasattr(stat_object, tag):
                    if tag == PLUGIN_OUTLIERS:
                        span.set_tag_str(span_path, getattr(stat_object, tag))
                        continue
                    span.set_tag(span_path, getattr(stat_object, tag))
