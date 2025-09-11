class DDTraceDeprecationWarning(DeprecationWarning):
    # Override module to simplify adding warning filters by querying for
    # ddtrace.DDTraceDeprecationWarning but not have to expose this in the
    # public API. This also allows us to avoid circular imports that would occur if
    # it was contained in the top-level ddtrace package.
    __module__ = "ddtrace"
