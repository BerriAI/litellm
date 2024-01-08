import logging

set_verbose = False


def print_verbose(print_statement):
    try:
        if set_verbose:
            print(print_statement)  # noqa
    except:
        pass


logging.basicConfig(level=logging.INFO)
# Create a custom logger for "debug-proxy"
debug_proxy_logger = logging.getLogger("LiteLLM Proxy")
