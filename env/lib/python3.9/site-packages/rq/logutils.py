import logging
import sys
from typing import Union

from rq.defaults import DEFAULT_LOGGING_DATE_FORMAT, DEFAULT_LOGGING_FORMAT


class _Colorizer:
    def __init__(self):
        esc = "\x1b["

        self.codes = {}
        self.codes[""] = ""
        self.codes["reset"] = esc + "39;49;00m"

        self.codes["bold"] = esc + "01m"
        self.codes["faint"] = esc + "02m"
        self.codes["standout"] = esc + "03m"
        self.codes["underline"] = esc + "04m"
        self.codes["blink"] = esc + "05m"
        self.codes["overline"] = esc + "06m"

        dark_colors = ["black", "darkred", "darkgreen", "brown", "darkblue", "purple", "teal", "lightgray"]
        light_colors = ["darkgray", "red", "green", "yellow", "blue", "fuchsia", "turquoise", "white"]

        x = 30
        for dark, light in zip(dark_colors, light_colors):
            self.codes[dark] = esc + "%im" % x
            self.codes[light] = esc + "%i;01m" % x
            x += 1

        del dark, light, x

        self.codes["darkteal"] = self.codes["turquoise"]
        self.codes["darkyellow"] = self.codes["brown"]
        self.codes["fuscia"] = self.codes["fuchsia"]
        self.codes["white"] = self.codes["bold"]

        if hasattr(sys.stdout, "isatty"):
            self.notty = not sys.stdout.isatty()
        else:
            self.notty = True

    def colorize(self, color_key, text):
        if self.notty:
            return text
        else:
            return self.codes[color_key] + text + self.codes["reset"]


colorizer = _Colorizer()


def make_colorizer(color: str):
    """Creates a function that colorizes text with the given color.

    For example::

        ..codeblock::python

            >>> green = make_colorizer('darkgreen')
            >>> red = make_colorizer('red')
            >>>
            >>> # You can then use:
            >>> print("It's either " + green('OK') + ' or ' + red('Oops'))
    """

    def inner(text):
        return colorizer.colorize(color, text)

    return inner


green = make_colorizer('darkgreen')
yellow = make_colorizer('darkyellow')
blue = make_colorizer('darkblue')
red = make_colorizer('darkred')


class ColorizingStreamHandler(logging.StreamHandler):
    levels = {
        logging.WARNING: yellow,
        logging.ERROR: red,
        logging.CRITICAL: red,
    }

    def __init__(self, exclude=None, *args, **kwargs):
        self.exclude = exclude
        super().__init__(*args, **kwargs)

    @property
    def is_tty(self):
        isatty = getattr(self.stream, 'isatty', None)
        return isatty and isatty()

    def format(self, record):
        message = logging.StreamHandler.format(self, record)
        if self.is_tty:
            # Don't colorize any traceback
            parts = message.split('\n', 1)
            parts[0] = " ".join([parts[0].split(" ", 1)[0], parts[0].split(" ", 1)[1]])

            message = '\n'.join(parts)

        return message


def setup_loghandlers(
    level: Union[int, str, None] = None,
    date_format: str = DEFAULT_LOGGING_DATE_FORMAT,
    log_format: str = DEFAULT_LOGGING_FORMAT,
    name: str = 'rq.worker',
):
    """Sets up a log handler.

    Args:
        level (Union[int, str, None], optional): The log level.
            Access an integer level (10-50) or a string level ("info", "debug" etc). Defaults to None.
        date_format (str, optional): The date format to use. Defaults to DEFAULT_LOGGING_DATE_FORMAT ('%H:%M:%S').
        log_format (str, optional): The log format to use.
            Defaults to DEFAULT_LOGGING_FORMAT ('%(asctime)s %(message)s').
        name (str, optional): The looger name. Defaults to 'rq.worker'.
    """
    logger = logging.getLogger(name)

    if not _has_effective_handler(logger):
        formatter = logging.Formatter(fmt=log_format, datefmt=date_format)
        handler = ColorizingStreamHandler(stream=sys.stdout)
        handler.setFormatter(formatter)
        handler.addFilter(lambda record: record.levelno < logging.ERROR)
        error_handler = ColorizingStreamHandler(stream=sys.stderr)
        error_handler.setFormatter(formatter)
        error_handler.addFilter(lambda record: record.levelno >= logging.ERROR)
        logger.addHandler(handler)
        logger.addHandler(error_handler)

    if level is not None:
        # The level may be a numeric value (e.g. when using the logging module constants)
        # Or a string representation of the logging level
        logger.setLevel(level if isinstance(level, int) else level.upper())


def _has_effective_handler(logger) -> bool:
    """
    Checks if a logger has a handler that will catch its messages in its logger hierarchy.

    Args:
        logger (logging.Logger): The logger to be checked.

    Returns:
        is_configured (bool): True if a handler is found for the logger, False otherwise.
    """
    while True:
        if logger.handlers:
            return True
        if not logger.parent:
            return False
        logger = logger.parent
