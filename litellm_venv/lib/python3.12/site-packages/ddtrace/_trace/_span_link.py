"""
Span Links
==========

Description
-----------

``ddtrace.trace.SpanLink`` introduces a new causal relationship between spans.
This new behavior is analogous to OpenTelemetry Span Links:
https://opentelemetry.io/docs/concepts/signals/traces/#span-links


Usage
-----

SpanLinks can be set using :meth:`ddtrace.Span.link_span(...)` Ex::

    from ddtrace import tracer

    s1 = tracer.trace("s1")
    s2 = tracer.trace("s2")

    link_attributes = {"link.name": "s1_to_s2", "link.kind": "scheduled_by", "key1": "val1"}
    s1.link_span(s2.context, link_attributes)
"""

import dataclasses
from enum import Enum
from typing import Optional

from ddtrace.internal.utils.formats import flatten_key_value


class SpanLinkKind(Enum):
    """
    A collection of standard SpanLink kinds. It's possible to use others, but these should be used when possible.
    """

    SPAN_POINTER = "span-pointer"  # Should not be used on normal SpanLinks.


def _id_not_zero(self, attribute_name, value):
    if not value > 0:
        raise ValueError(f"{attribute_name} must be > 0. Value is {value}")


@dataclasses.dataclass
class SpanLink:
    """
    TraceId [required]: The span's 128-bit Trace ID
    SpanId [required]: The span's 64-bit Span ID

    Flags [optional]: The span's trace-flags field, as defined in the W3C standard. If only sampling
    information is provided, the flags value must be 1 if the decision is keep, otherwise 0.

    TraceState [optional]: The span's tracestate field, as defined in the W3C standard.

    Attributes [optional]: Zero or more key-value pairs, where the key must be a non-empty string and the
    value is either a string, bool, number or an array of primitive type values.
    """

    trace_id: int
    span_id: int
    tracestate: Optional[str] = None
    flags: Optional[int] = None
    attributes: dict = dataclasses.field(default_factory=dict)
    _dropped_attributes: int = 0

    def __post_init__(self):
        _id_not_zero(self, "trace_id", self.trace_id)
        _id_not_zero(self, "span_id", self.span_id)

    @property
    def name(self):
        return self.attributes["link.name"]

    @name.setter
    def name(self, value):
        self.attributes["link.name"] = value

    @property
    def kind(self) -> Optional[str]:
        return self.attributes.get("link.kind")

    @kind.setter
    def kind(self, value: str) -> None:
        self.attributes["link.kind"] = value

    def _drop_attribute(self, key):
        if key not in self.attributes:
            raise KeyError(f"Invalid key: {key}")
        del self.attributes[key]
        self._dropped_attributes += 1

    def to_dict(self):
        d = {
            "trace_id": "{:032x}".format(self.trace_id),
            "span_id": "{:016x}".format(self.span_id),
        }
        if self.attributes:
            d["attributes"] = {}
            for k, v in self.attributes.items():
                # flatten all values with the type list, tuple and set
                for k1, v1 in flatten_key_value(k, v).items():
                    # convert all values to string
                    if isinstance(v1, str):
                        d["attributes"][k1] = v1
                    elif isinstance(v1, bool):
                        # convert bool to lowercase string to be consistent with json encoding
                        d["attributes"][k1] = str(v1).lower()
                    else:
                        d["attributes"][k1] = str(v1)

        if self._dropped_attributes > 0:
            d["dropped_attributes_count"] = self._dropped_attributes
        if self.tracestate:
            d["tracestate"] = self.tracestate
        if self.flags is not None:
            d["flags"] = self.flags

        return d

    def __str__(self) -> str:
        attrs_str = ",".join([f"{k}:{v}" for k, v in self.attributes.items()])
        return (
            f"trace_id={self.trace_id} span_id={self.span_id} attributes={attrs_str} "
            f"tracestate={self.tracestate} flags={self.flags} dropped_attributes={self._dropped_attributes}"
        )
