from typing import List  # noqa:F401
from typing import Tuple  # noqa:F401

from ...constants import ENV_KEY
from ...constants import VERSION_KEY
from .collector import ValueCollector
from .constants import LANG
from .constants import LANG_INTERPRETER
from .constants import LANG_VERSION
from .constants import SERVICE
from .constants import TRACER_VERSION


class RuntimeTagCollector(ValueCollector):
    periodic = False
    value = []  # type: List[Tuple[str, str]]


class TracerTagCollector(RuntimeTagCollector):
    """Tag collector for the ddtrace Tracer"""

    required_modules = ["ddtrace"]

    def collect_fn(self, keys):
        ddtrace = self.modules.get("ddtrace")

        # make sure to copy _services to avoid RuntimeError: Set changed size during iteration
        tags = [(SERVICE, service) for service in list(ddtrace.tracer._services)]

        # DEV: `DD_ENV`, `DD_VERSION`, and `DD_SERVICE` get picked up automatically by
        #      dogstatsd client, but someone might configure these via `ddtrace.config`
        #      instead of env vars, so better to collect them here again just in case
        # DD_ENV gets stored in `config.env`
        if ddtrace.config.env:
            tags.append((ENV_KEY, ddtrace.config.env))

        # DD_VERSION gets stored in `config.version`
        if ddtrace.config.version:
            tags.append((VERSION_KEY, ddtrace.config.version))

        for key, value in ddtrace.tracer._tags.items():
            tags.append((key, value))
        return tags


class PlatformTagCollector(RuntimeTagCollector):
    """Tag collector for the Python interpreter implementation.

    Tags collected:
    - ``lang_interpreter``:

      * For CPython this is 'CPython'.
      * For Pypy this is ``PyPy``
      * For Jython this is ``Jython``

    - `lang_version``,  eg ``2.7.10``
    - ``lang`` e.g. ``Python``
    - ``tracer_version`` e.g. ``0.29.0``

    """

    required_modules = ["platform", "ddtrace"]

    def collect_fn(self, keys):
        platform = self.modules.get("platform")
        ddtrace = self.modules.get("ddtrace")
        tags = [
            (LANG, "python"),
            (LANG_INTERPRETER, platform.python_implementation()),
            (LANG_VERSION, platform.python_version()),
            (TRACER_VERSION, ddtrace.__version__),
        ]
        return tags
