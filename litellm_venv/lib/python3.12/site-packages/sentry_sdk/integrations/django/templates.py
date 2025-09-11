import functools

from django.template import TemplateSyntaxError
from django.utils.safestring import mark_safe
from django import VERSION as DJANGO_VERSION

import sentry_sdk
from sentry_sdk.consts import OP
from sentry_sdk.utils import ensure_integration_enabled

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
    from typing import Dict
    from typing import Optional
    from typing import Iterator
    from typing import Tuple

try:
    # support Django 1.9
    from django.template.base import Origin
except ImportError:
    # backward compatibility
    from django.template.loader import LoaderOrigin as Origin


def get_template_frame_from_exception(exc_value):
    # type: (Optional[BaseException]) -> Optional[Dict[str, Any]]

    # As of Django 1.9 or so the new template debug thing showed up.
    if hasattr(exc_value, "template_debug"):
        return _get_template_frame_from_debug(exc_value.template_debug)  # type: ignore

    # As of r16833 (Django) all exceptions may contain a
    # ``django_template_source`` attribute (rather than the legacy
    # ``TemplateSyntaxError.source`` check)
    if hasattr(exc_value, "django_template_source"):
        return _get_template_frame_from_source(
            exc_value.django_template_source  # type: ignore
        )

    if isinstance(exc_value, TemplateSyntaxError) and hasattr(exc_value, "source"):
        source = exc_value.source
        if isinstance(source, (tuple, list)) and isinstance(source[0], Origin):
            return _get_template_frame_from_source(source)  # type: ignore

    return None


def _get_template_name_description(template_name):
    # type: (str) -> str
    if isinstance(template_name, (list, tuple)):
        if template_name:
            return "[{}, ...]".format(template_name[0])
    else:
        return template_name


def patch_templates():
    # type: () -> None
    from django.template.response import SimpleTemplateResponse
    from sentry_sdk.integrations.django import DjangoIntegration

    real_rendered_content = SimpleTemplateResponse.rendered_content

    @property  # type: ignore
    @ensure_integration_enabled(DjangoIntegration, real_rendered_content.fget)
    def rendered_content(self):
        # type: (SimpleTemplateResponse) -> str
        with sentry_sdk.start_span(
            op=OP.TEMPLATE_RENDER,
            name=_get_template_name_description(self.template_name),
            origin=DjangoIntegration.origin,
        ) as span:
            span.set_data("context", self.context_data)
            return real_rendered_content.fget(self)

    SimpleTemplateResponse.rendered_content = rendered_content

    if DJANGO_VERSION < (1, 7):
        return
    import django.shortcuts

    real_render = django.shortcuts.render

    @functools.wraps(real_render)
    @ensure_integration_enabled(DjangoIntegration, real_render)
    def render(request, template_name, context=None, *args, **kwargs):
        # type: (django.http.HttpRequest, str, Optional[Dict[str, Any]], *Any, **Any) -> django.http.HttpResponse

        # Inject trace meta tags into template context
        context = context or {}
        if "sentry_trace_meta" not in context:
            context["sentry_trace_meta"] = mark_safe(
                sentry_sdk.get_current_scope().trace_propagation_meta()
            )

        with sentry_sdk.start_span(
            op=OP.TEMPLATE_RENDER,
            name=_get_template_name_description(template_name),
            origin=DjangoIntegration.origin,
        ) as span:
            span.set_data("context", context)
            return real_render(request, template_name, context, *args, **kwargs)

    django.shortcuts.render = render


def _get_template_frame_from_debug(debug):
    # type: (Dict[str, Any]) -> Dict[str, Any]
    if debug is None:
        return None

    lineno = debug["line"]
    filename = debug["name"]
    if filename is None:
        filename = "<django template>"

    pre_context = []
    post_context = []
    context_line = None

    for i, line in debug["source_lines"]:
        if i < lineno:
            pre_context.append(line)
        elif i > lineno:
            post_context.append(line)
        else:
            context_line = line

    return {
        "filename": filename,
        "lineno": lineno,
        "pre_context": pre_context[-5:],
        "post_context": post_context[:5],
        "context_line": context_line,
        "in_app": True,
    }


def _linebreak_iter(template_source):
    # type: (str) -> Iterator[int]
    yield 0
    p = template_source.find("\n")
    while p >= 0:
        yield p + 1
        p = template_source.find("\n", p + 1)


def _get_template_frame_from_source(source):
    # type: (Tuple[Origin, Tuple[int, int]]) -> Optional[Dict[str, Any]]
    if not source:
        return None

    origin, (start, end) = source
    filename = getattr(origin, "loadname", None)
    if filename is None:
        filename = "<django template>"
    template_source = origin.reload()
    lineno = None
    upto = 0
    pre_context = []
    post_context = []
    context_line = None

    for num, next in enumerate(_linebreak_iter(template_source)):
        line = template_source[upto:next]
        if start >= upto and end <= next:
            lineno = num
            context_line = line
        elif lineno is None:
            pre_context.append(line)
        else:
            post_context.append(line)

        upto = next

    if context_line is None or lineno is None:
        return None

    return {
        "filename": filename,
        "lineno": lineno,
        "pre_context": pre_context[-5:],
        "post_context": post_context[:5],
        "context_line": context_line,
    }
