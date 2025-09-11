# See ../ddup/__init__.py for some discussion on the is_available attribute.
# The configuration for this feature is handled in ddtrace/settings/crashtracker.py.
is_available = False
failure_msg = ""


def _default_return_false(*args, **kwargs):
    return False


try:
    from ._crashtracker import *  # noqa: F403, F401

    is_available = True

except Exception as e:
    failure_msg = str(e)

    # Crashtracker is used early during startup, and so it must be robust across installations.
    # Here we just stub everything.
    def __getattr__(name):
        if name == "failure_msg":
            return failure_msg
        if name == "is_available":
            return False
        return _default_return_false
