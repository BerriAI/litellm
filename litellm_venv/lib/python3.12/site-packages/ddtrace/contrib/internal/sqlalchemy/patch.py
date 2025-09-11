import sqlalchemy
from wrapt import wrap_function_wrapper as _w

from ddtrace.appsec._iast._metrics import _set_metric_iast_instrumented_sink
from ddtrace.appsec._iast.constants import VULN_SQL_INJECTION
from ddtrace.contrib.trace_utils import unwrap
from ddtrace.settings.asm import config as asm_config

from .engine import _wrap_create_engine


def get_version():
    # type: () -> str
    return getattr(sqlalchemy, "__version__", "")


def patch():
    if getattr(sqlalchemy.engine, "__datadog_patch", False):
        return
    sqlalchemy.engine.__datadog_patch = True

    # patch the engine creation function
    _w("sqlalchemy", "create_engine", _wrap_create_engine)
    _w("sqlalchemy.engine", "create_engine", _wrap_create_engine)

    if asm_config._iast_enabled:
        _set_metric_iast_instrumented_sink(VULN_SQL_INJECTION)


def unpatch():
    # unpatch sqlalchemy
    if getattr(sqlalchemy.engine, "__datadog_patch", False):
        sqlalchemy.engine.__datadog_patch = False
        unwrap(sqlalchemy, "create_engine")
        unwrap(sqlalchemy.engine, "create_engine")
