import logging

set_verbose = False

# Create a handler for the logger (you may need to adapt this based on your needs)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

# Create a formatter and set it for the handler

formatter = logging.Formatter("\033[92m%(name)s - %(levelname)s\033[0m: %(message)s")

handler.setFormatter(formatter)

verbose_proxy_logger = logging.getLogger("LiteLLM Proxy")
verbose_router_logger = logging.getLogger("LiteLLM Router")
verbose_logger = logging.getLogger("LiteLLM")

# Add the handler to the logger
verbose_router_logger.addHandler(handler)
verbose_proxy_logger.addHandler(handler)
verbose_logger.addHandler(handler)


def _turn_on_debug():
    verbose_logger.setLevel(level=logging.DEBUG)  # set package log to debug
    verbose_router_logger.setLevel(level=logging.DEBUG)  # set router logs to debug
    verbose_proxy_logger.setLevel(level=logging.DEBUG)  # set proxy logs to debug


def print_verbose(print_statement):
    try:
        if set_verbose:
            print(print_statement)  # noqa
    except:
        pass
