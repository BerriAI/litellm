BENCHMARK_INFO = "benchmark.duration.info"
BENCHMARK_MEAN = "benchmark.duration.mean"
BENCHMARK_RUN = "benchmark.duration.runs"

STATISTICS_HD15IQR = "benchmark.duration.statistics.hd15iqr"
STATISTICS_IQR = "benchmark.duration.statistics.iqr"
STATISTICS_IQR_OUTLIERS = "benchmark.duration.statistics.iqr_outliers"
STATISTICS_LD15IQR = "benchmark.duration.statistics.ld15iqr"
STATISTICS_MAX = "benchmark.duration.statistics.max"
STATISTICS_MEAN = "benchmark.duration.statistics.mean"
STATISTICS_MEDIAN = "benchmark.duration.statistics.median"
STATISTICS_MIN = "benchmark.duration.statistics.min"
STATISTICS_N = "benchmark.duration.statistics.n"
STATISTICS_OPS = "benchmark.duration.statistics.ops"
STATISTICS_OUTLIERS = "benchmark.duration.statistics.outliers"
STATISTICS_Q1 = "benchmark.duration.statistics.q1"
STATISTICS_Q3 = "benchmark.duration.statistics.q3"
STATISTICS_STDDEV = "benchmark.duration.statistics.std_dev"
STATISTICS_STDDEV_OUTLIERS = "benchmark.duration.statistics.std_dev_outliers"
STATISTICS_TOTAL = "benchmark.duration.statistics.total"

PLUGIN_HD15IQR = "hd15iqr"
PLUGIN_IQR = "iqr"
PLUGIN_IQR_OUTLIERS = "iqr_outliers"
PLUGIN_LD15IQR = "ld15iqr"
PLUGIN_MAX = "max"
PLUGIN_MEAN = "mean"
PLUGIN_MEDIAN = "median"
PLUGIN_MIN = "min"
PLUGIN_OPS = "ops"
PLUGIN_OUTLIERS = "outliers"
PLUGIN_Q1 = "q1"
PLUGIN_Q3 = "q3"
PLUGIN_ROUNDS = "rounds"
PLUGIN_STDDEV = "stddev"
PLUGIN_STDDEV_OUTLIERS = "stddev_outliers"
PLUGIN_TOTAL = "total"

PLUGIN_METRICS = {
    BENCHMARK_MEAN: PLUGIN_MEAN,
    BENCHMARK_RUN: PLUGIN_ROUNDS,
    STATISTICS_HD15IQR: PLUGIN_HD15IQR,
    STATISTICS_IQR: PLUGIN_IQR,
    STATISTICS_IQR_OUTLIERS: PLUGIN_IQR_OUTLIERS,
    STATISTICS_LD15IQR: PLUGIN_LD15IQR,
    STATISTICS_MAX: PLUGIN_MAX,
    STATISTICS_MEAN: PLUGIN_MEAN,
    STATISTICS_MEDIAN: PLUGIN_MEDIAN,
    STATISTICS_MIN: PLUGIN_MIN,
    STATISTICS_OPS: PLUGIN_OPS,
    STATISTICS_OUTLIERS: PLUGIN_OUTLIERS,
    STATISTICS_Q1: PLUGIN_Q1,
    STATISTICS_Q3: PLUGIN_Q3,
    STATISTICS_N: PLUGIN_ROUNDS,
    STATISTICS_STDDEV: PLUGIN_STDDEV,
    STATISTICS_STDDEV_OUTLIERS: PLUGIN_STDDEV_OUTLIERS,
    STATISTICS_TOTAL: PLUGIN_TOTAL,
}

PLUGIN_METRICS_V2 = {
    "duration_mean": PLUGIN_MEAN,
    "duration_runs": PLUGIN_ROUNDS,
    "statistics_hd15iqr": PLUGIN_HD15IQR,
    "statistics_iqr": PLUGIN_IQR,
    "statistics_iqr_outliers": PLUGIN_IQR_OUTLIERS,
    "statistics_ld15iqr": PLUGIN_LD15IQR,
    "statistics_max": PLUGIN_MAX,
    "statistics_mean": PLUGIN_MEAN,
    "statistics_median": PLUGIN_MEDIAN,
    "statistics_min": PLUGIN_MIN,
    "statistics_n": PLUGIN_ROUNDS,
    "statistics_ops": PLUGIN_OPS,
    "statistics_outliers": PLUGIN_OUTLIERS,
    "statistics_q1": PLUGIN_Q1,
    "statistics_q3": PLUGIN_Q3,
    "statistics_std_dev": PLUGIN_STDDEV,
    "statistics_std_dev_outliers": PLUGIN_STDDEV_OUTLIERS,
    "statistics_total": PLUGIN_TOTAL,
}
