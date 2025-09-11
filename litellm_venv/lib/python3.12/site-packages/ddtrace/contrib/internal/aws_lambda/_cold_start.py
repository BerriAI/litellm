__cold_start = True
__lambda_container_initialized = False


def set_cold_start():
    """Set the value of the cold start global.

    This should be executed once per AWS Lambda execution before the execution.
    """
    global __cold_start
    global __lambda_container_initialized
    __cold_start = not __lambda_container_initialized
    __lambda_container_initialized = True


def is_cold_start():
    """Returns the value of the global cold_start."""
    return __cold_start
