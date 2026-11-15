from .types import CacheablePrefix, CacheCounters


def write_proves_prefix_eligible(prefix: CacheablePrefix, counters: CacheCounters) -> bool:
    """Only a cache write on a request with a single breakpoint proves this prefix met the minimum. With
    several breakpoints the counters are a total over all of them, so an earlier, shorter prefix can account
    for every token, and a read can be served from an earlier entry without this prefix ever being cached."""
    return prefix.breakpoints == 1 and counters.write_tokens > 0
