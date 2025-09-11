import typing as t

from envier import En

from ddtrace.internal.utils.formats import parse_tags_str
from ddtrace.settings._core import report_telemetry as _report_telemetry


resolver_default = "full"


def _derive_stacktrace_resolver(config: "CrashtrackingConfig") -> t.Optional[str]:
    resolver = str(config._stacktrace_resolver or "")
    resolver = resolver.lower()
    if resolver == "none":
        return None
    if resolver in ("fast", "full", "safe"):
        return resolver

    # Invalid values should degrade to the default
    return resolver_default


def _check_for_crashtracking_available() -> bool:
    from ddtrace.internal.datadog.profiling import crashtracker

    return crashtracker.is_available


def _derive_crashtracking_enabled(config: "CrashtrackingConfig") -> bool:
    if not _check_for_crashtracking_available():
        return False
    return bool(config._enabled)


class CrashtrackingConfig(En):
    # Although the component is called crashtrack_er_, for consistency with other products/telemetry we use the term
    # crashtrack_ing_ as much as possible.  We'll gradually align on this.
    __prefix__ = "dd.crashtracking"

    _enabled = En.v(
        bool,
        "enabled",
        default=True,
        help_type="Boolean",
        help="Enables crashtracking",
    )

    enabled = En.d(bool, _derive_crashtracking_enabled)

    debug_url = En.v(
        t.Optional[str],
        "debug_url",
        default=None,
        help_type="String",
        help="Overrides the URL parameter set by the ddtrace library. "
        "This is generally useful only for dd-trace-py development.",
    )

    stdout_filename = En.v(
        t.Optional[str],
        "stdout_filename",
        default=None,
        help_type="String",
        help="The destination filename for crashtracking stdout",
    )

    stderr_filename = En.v(
        t.Optional[str],
        "stderr_filename",
        default=None,
        help_type="String",
        help="The destination filename for crashtracking stderr",
    )

    use_alt_stack = En.v(
        bool,
        "use_alt_stack",
        default=True,
        help_type="Boolean",
        help="Whether to use an alternate stack for crashtracking.",
    )

    create_alt_stack = En.v(
        bool,
        "create_alt_stack",
        default=True,
        help_type="Boolean",
        help="Whether to create an alternate stack for crashtracking."
        "The Python runtime creates an altstack of very small size, so this parameter is typically combined with"
        "use_alt_stack to ensure that the altstack is large enough.",
    )

    _stacktrace_resolver = En.v(
        t.Optional[str],
        "stacktrace_resolver",
        default=resolver_default,
        help_type="String",
        help="How to collect native stack traces during a crash, if at all.  Accepted values are 'none', 'fast',"
        " 'safe', and 'full'.  The default value is '" + resolver_default + "'.",
    )
    stacktrace_resolver = En.d(t.Optional[str], _derive_stacktrace_resolver)

    tags = En.v(
        dict,
        "tags",
        parser=parse_tags_str,
        default={},
        help_type="Mapping",
        help="Additional crashtracking tags. Must be a list in the ``key1:value,key2:value2`` format. "
        "This is generally useful only for dd-trace-py development.",
    )

    wait_for_receiver = En.v(
        bool,
        "wait_for_receiver",
        default=True,
        help_type="Boolean",
        help="Whether to wait for the crashtracking receiver",
    )


config = CrashtrackingConfig()
_report_telemetry(config)
