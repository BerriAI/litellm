from typing import Optional  # noqa:F401

from ddtrace.internal.utils.cache import cached


@cached()
def get_wsgi_header(header):
    # type: (str) -> str
    """Returns a WSGI compliant HTTP header.
    See https://www.python.org/dev/peps/pep-3333/#environ-variables for
    information from the spec.
    """
    return "HTTP_{}".format(header.upper().replace("-", "_"))


@cached()
def from_wsgi_header(header):
    # type: (str) -> Optional[str]
    """Convert a WSGI compliant HTTP header into the original header.
    See https://www.python.org/dev/peps/pep-3333/#environ-variables for
    information from the spec.
    """
    HTTP_PREFIX = "HTTP_"
    # PEP 333 gives two headers which aren't prepended with HTTP_.
    UNPREFIXED_HEADERS = {"CONTENT_TYPE", "CONTENT_LENGTH"}

    if header.startswith(HTTP_PREFIX):
        header = header[len(HTTP_PREFIX) :]
    elif header not in UNPREFIXED_HEADERS:
        return None
    return header.replace("_", "-").title()
