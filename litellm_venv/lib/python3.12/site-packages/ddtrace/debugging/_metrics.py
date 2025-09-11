from ddtrace.internal.metrics import Metrics


# Debugger metrics
metrics = Metrics(namespace="debugger")

# Metric probe metrics (always enabled)
probe_metrics = Metrics(namespace="debugger.metric")
probe_metrics.enable()
