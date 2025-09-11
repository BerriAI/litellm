import os
import sys
import typing  # noqa:F401


def get_application_name():
    # type: () -> typing.Optional[str]
    """Attempts to find the application name using system arguments."""
    try:
        import __main__

        name = __main__.__file__
    except (ImportError, AttributeError):
        try:
            name = sys.argv[0]
        except (AttributeError, IndexError):
            return None

    return os.path.basename(name)
