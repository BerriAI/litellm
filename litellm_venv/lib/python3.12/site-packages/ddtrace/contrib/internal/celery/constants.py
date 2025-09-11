from ddtrace import config


# Celery Context key
SPAN_KEY = "__dd_task_span"
CTX_KEY = "__dd_task_context"

# Span names
PRODUCER_ROOT_SPAN = "celery.apply"
WORKER_ROOT_SPAN = "celery.run"

# Task operations
TASK_TAG_KEY = "celery.action"
TASK_APPLY = "apply"
TASK_APPLY_ASYNC = "apply_async"
TASK_RUN = "run"
TASK_RETRY_REASON_KEY = "celery.retry.reason"

# Service info
PRODUCER_SERVICE = config._get_service(default="celery-producer")
WORKER_SERVICE = config._get_service(default="celery-worker")
