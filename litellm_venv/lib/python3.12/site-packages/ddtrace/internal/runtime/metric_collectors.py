import os
from typing import List  # noqa:F401
from typing import Tuple  # noqa:F401

from .collector import ValueCollector
from .constants import CPU_PERCENT
from .constants import CPU_TIME_SYS
from .constants import CPU_TIME_USER
from .constants import CTX_SWITCH_INVOLUNTARY
from .constants import CTX_SWITCH_VOLUNTARY
from .constants import GC_COUNT_GEN0
from .constants import GC_COUNT_GEN1
from .constants import GC_COUNT_GEN2
from .constants import MEM_RSS
from .constants import THREAD_COUNT


class RuntimeMetricCollector(ValueCollector):
    value = []  # type: List[Tuple[str, str]]
    periodic = True


class GCRuntimeMetricCollector(RuntimeMetricCollector):
    """Collector for garbage collection generational counts

    More information at https://docs.python.org/3/library/gc.html
    """

    required_modules = ["gc"]

    def collect_fn(self, keys):
        gc = self.modules.get("gc")

        counts = gc.get_count()
        metrics = [
            (GC_COUNT_GEN0, counts[0]),
            (GC_COUNT_GEN1, counts[1]),
            (GC_COUNT_GEN2, counts[2]),
        ]

        return metrics


class PSUtilRuntimeMetricCollector(RuntimeMetricCollector):
    """Collector for psutil metrics.

    Performs batched operations via proc.oneshot() to optimize the calls.
    See https://psutil.readthedocs.io/en/latest/#psutil.Process.oneshot
    for more information.
    """

    required_modules = ["ddtrace.vendor.psutil"]
    delta_funs = {
        CPU_TIME_SYS: lambda p: p.cpu_times().system,
        CPU_TIME_USER: lambda p: p.cpu_times().user,
        CTX_SWITCH_VOLUNTARY: lambda p: p.num_ctx_switches().voluntary,
        CTX_SWITCH_INVOLUNTARY: lambda p: p.num_ctx_switches().involuntary,
    }
    abs_funs = {
        THREAD_COUNT: lambda p: p.num_threads(),
        MEM_RSS: lambda p: p.memory_info().rss,
        CPU_PERCENT: lambda p: p.cpu_percent(),
    }

    def _on_modules_load(self):
        self.proc = self.modules["ddtrace.vendor.psutil"].Process(os.getpid())
        self.stored_values = {key: 0 for key in self.delta_funs.keys()}

    def collect_fn(self, keys):
        with self.proc.oneshot():
            metrics = {}

            # Populate metrics for which we compute delta values
            for metric, delta_fun in self.delta_funs.items():
                try:
                    value = delta_fun(self.proc)
                except Exception:
                    value = 0

                delta = value - self.stored_values.get(metric, 0)
                self.stored_values[metric] = value
                metrics[metric] = delta

            # Populate metrics that just take instantaneous reading
            for metric, abs_fun in self.abs_funs.items():
                try:
                    value = abs_fun(self.proc)
                except Exception:
                    value = 0

                metrics[metric] = value

            return list(metrics.items())
