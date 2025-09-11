import pytest

from ddtrace.contrib.pytest._utils import _get_test_id_from_item
from ddtrace.contrib.pytest_benchmark.constants import PLUGIN_METRICS_V2
from ddtrace.internal.logger import get_logger
from ddtrace.internal.test_visibility._benchmark_mixin import BenchmarkDurationData
from ddtrace.internal.test_visibility.api import InternalTest


log = get_logger(__name__)


def _set_benchmark_data_from_item(item: pytest.Item) -> None:
    try:
        fixture = hasattr(item, "funcargs") and item.funcargs.get("benchmark")

        if not fixture or not fixture.stats:
            return

        stat_object = item.funcargs.get("benchmark").stats.stats

        data_kwargs = {}

        for data_attr, stats_attr in PLUGIN_METRICS_V2.items():
            if hasattr(stat_object, stats_attr):
                data_kwargs[data_attr] = getattr(stat_object, stats_attr)

        test_id = _get_test_id_from_item(item)
        benchmark_data = BenchmarkDurationData(**data_kwargs)

        InternalTest.set_benchmark_data(test_id, benchmark_data, is_benchmark=True)

    except Exception:  # noqa: E722
        log.debug("Unable to set benchmark data for item %s", item, exc_info=True)
        return None
