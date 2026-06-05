from . import *


def __getattr__(name):
    # ``litellm.proxy.auth`` moved to ``litellm.auth``. Resolve the attribute
    # lazily so legacy ``litellm.proxy.auth`` access (and mock.patch targets)
    # keeps working even though nothing imports the compatibility shim directly.
    if name == "auth":
        import litellm.proxy.auth as _auth

        return _auth
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
