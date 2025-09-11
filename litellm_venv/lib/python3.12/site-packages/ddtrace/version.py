def get_version():
    # type: () -> str
    try:
        from ._version import version

        return version
    except ImportError:
        try:
            from importlib.metadata import version as ilm_version
        except ImportError:
            # required for python3.7
            from importlib_metadata import version as ilm_version  # type: ignore[no-redef]
        try:
            return ilm_version("ddtrace")
        except ModuleNotFoundError:
            # package is not installed
            return "dev"
