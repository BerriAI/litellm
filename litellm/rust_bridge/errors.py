class RustRouteUnavailableError(RuntimeError):
    pass


class RustRouteDeclinedError(RuntimeError):
    pass


class RustRouteUnsupportedError(NotImplementedError):
    pass
