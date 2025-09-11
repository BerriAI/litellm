from enum import Enum


TELEMETRY_NAMESPACE_TAG_TRACER = "tracers"
TELEMETRY_NAMESPACE_TAG_APPSEC = "appsec"
TELEMETRY_NAMESPACE_TAG_IAST = "iast"

TELEMETRY_TYPE_GENERATE_METRICS = "generate-metrics"
TELEMETRY_TYPE_DISTRIBUTION = "distributions"
TELEMETRY_TYPE_LOGS = "logs"


class TELEMETRY_LOG_LEVEL(Enum):
    DEBUG = "DEBUG"
    WARNING = "WARN"
    ERROR = "ERROR"


class TELEMETRY_APM_PRODUCT(str, Enum):
    LLMOBS = "mlobs"
    DYNAMIC_INSTRUMENTATION = "dynamic_instrumentation"
    PROFILER = "profiler"
    APPSEC = "appsec"
