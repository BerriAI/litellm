import re

from ddtrace.ext import elasticsearch as metadata


# Replace any ID
ID_REGEXP = re.compile(r"/([0-9]+)([/\?]|$)")
ID_PLACEHOLDER = r"/?\2"

# Remove digits from potential timestamped indexes (should be an option).
# For now, let's say 2+ digits
INDEX_REGEXP = re.compile(r"[0-9]{2,}")
INDEX_PLACEHOLDER = r"?"


def quantize(span):
    """Quantize an elasticsearch span

    We want to extract a meaningful `resource` from the request.
    We do it based on the method + url, with some cleanup applied to the URL.

    The URL might a ID, but also it is common to have timestamped indexes.
    While the first is easy to catch, the second should probably be configurable.

    All of this should probably be done in the Agent. Later.
    """
    url = span.get_tag(metadata.URL)
    method = span.get_tag(metadata.METHOD)

    quantized_url = ID_REGEXP.sub(ID_PLACEHOLDER, url)
    quantized_url = INDEX_REGEXP.sub(INDEX_PLACEHOLDER, quantized_url)

    span.resource = "{method} {url}".format(method=method, url=quantized_url)

    return span
