from ddtrace.appsec._iast._handlers import _on_django_func_wrapped
from ddtrace.appsec._iast._handlers import _on_django_patch
from ddtrace.appsec._iast._handlers import _on_flask_patch
from ddtrace.appsec._iast._handlers import _on_grpc_response
from ddtrace.appsec._iast._handlers import _on_request_init
from ddtrace.appsec._iast._handlers import _on_set_http_meta_iast
from ddtrace.appsec._iast._handlers import _on_wsgi_environ
from ddtrace.appsec._iast._iast_request_context import _iast_end_request
from ddtrace.internal import core


def iast_listen():
    core.on("grpc.client.response.message", _on_grpc_response)
    core.on("grpc.server.response.message", _on_grpc_server_response)

    core.on("set_http_meta_for_asm", _on_set_http_meta_iast)
    core.on("django.patch", _on_django_patch)
    core.on("django.wsgi_environ", _on_wsgi_environ, "wrapped_result")
    core.on("django.func.wrapped", _on_django_func_wrapped)
    core.on("flask.patch", _on_flask_patch)
    core.on("flask.request_init", _on_request_init)

    core.on("context.ended.wsgi.__call__", _iast_end_request)
    core.on("context.ended.asgi.__call__", _iast_end_request)


def _on_grpc_server_response(message):
    _on_grpc_response(message)
