import dogpile

from ddtrace.constants import SPAN_MEASURED_KEY
from ddtrace.ext import SpanTypes
from ddtrace.ext import db
from ddtrace.internal.constants import COMPONENT
from ddtrace.internal.schema import schematize_cache_operation
from ddtrace.internal.schema import schematize_service_name
from ddtrace.internal.utils import get_argument_value
from ddtrace.pin import Pin


def _wrap_get_create(func, instance, args, kwargs):
    pin = Pin.get_from(dogpile.cache)
    if not pin or not pin.enabled():
        return func(*args, **kwargs)

    key = get_argument_value(args, kwargs, 0, "key")
    with pin.tracer.trace(
        schematize_cache_operation("dogpile.cache", cache_provider="dogpile"),
        service=schematize_service_name(None),
        resource="get_or_create",
        span_type=SpanTypes.CACHE,
    ) as span:
        span.set_tag_str(COMPONENT, "dogpile_cache")
        span.set_tag(SPAN_MEASURED_KEY)
        span.set_tag("key", key)
        span.set_tag("region", instance.name)
        span.set_tag("backend", instance.actual_backend.__class__.__name__)
        response = func(*args, **kwargs)
        span.set_metric(db.ROWCOUNT, 1)
        return response


def _wrap_get_create_multi(func, instance, args, kwargs):
    pin = Pin.get_from(dogpile.cache)
    if not pin or not pin.enabled():
        return func(*args, **kwargs)

    keys = get_argument_value(args, kwargs, 0, "keys")
    with pin.tracer.trace(
        schematize_cache_operation("dogpile.cache", cache_provider="dogpile"),
        service=schematize_service_name(None),
        resource="get_or_create_multi",
        span_type="cache",
    ) as span:
        span.set_tag_str(COMPONENT, "dogpile_cache")
        span.set_tag(SPAN_MEASURED_KEY)
        span.set_tag("keys", keys)
        span.set_tag("region", instance.name)
        span.set_tag("backend", instance.actual_backend.__class__.__name__)
        response = func(*args, **kwargs)
        span.set_metric(db.ROWCOUNT, len(response))
        return response
