from ddtrace.constants import AUTO_KEEP
from ddtrace.constants import AUTO_REJECT
from ddtrace.constants import USER_KEEP
from ddtrace.constants import USER_REJECT


PROPAGATION_STYLE_DATADOG = "datadog"
PROPAGATION_STYLE_B3_MULTI = "b3multi"
PROPAGATION_STYLE_B3_SINGLE = "b3"
_PROPAGATION_STYLE_W3C_TRACECONTEXT = "tracecontext"
_PROPAGATION_STYLE_NONE = "none"
_PROPAGATION_STYLE_DEFAULT = "datadog,tracecontext,baggage"
_PROPAGATION_STYLE_BAGGAGE = "baggage"
PROPAGATION_STYLE_ALL = (
    _PROPAGATION_STYLE_W3C_TRACECONTEXT,
    PROPAGATION_STYLE_DATADOG,
    PROPAGATION_STYLE_B3_MULTI,
    PROPAGATION_STYLE_B3_SINGLE,
    _PROPAGATION_STYLE_NONE,
    _PROPAGATION_STYLE_BAGGAGE,
)
W3C_TRACESTATE_KEY = "tracestate"
W3C_TRACEPARENT_KEY = "traceparent"
W3C_TRACESTATE_PARENT_ID_KEY = "p"
W3C_TRACESTATE_ORIGIN_KEY = "o"
W3C_TRACESTATE_SAMPLING_PRIORITY_KEY = "s"
DEFAULT_SAMPLING_RATE_LIMIT = 100
SAMPLING_DECISION_TRACE_TAG_KEY = "_dd.p.dm"
LAST_DD_PARENT_ID_KEY = "_dd.parent_id"
DEFAULT_SERVICE_NAME = "unnamed-python-service"
# Used to set the name of an integration on a span
COMPONENT = "component"
HIGHER_ORDER_TRACE_ID_BITS = "_dd.p.tid"
MAX_UINT_64BITS = (1 << 64) - 1
SPAN_LINKS_KEY = "_dd.span_links"
SPAN_EVENTS_KEY = "events"
SPAN_API_DATADOG = "datadog"
SPAN_API_OTEL = "otel"
SPAN_API_OPENTRACING = "opentracing"
DEFAULT_BUFFER_SIZE = 20 << 20  # 20 MB
DEFAULT_MAX_PAYLOAD_SIZE = 20 << 20  # 20 MB
DEFAULT_PROCESSING_INTERVAL = 1.0
DEFAULT_REUSE_CONNECTIONS = False
BLOCKED_RESPONSE_HTML = """<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>You've been blocked</title><style>a,body,div,html,span{margin:0;padding:0;border:0;font-size:100%;font:inherit;vertical-align:baseline}body{background:-webkit-radial-gradient(26% 19%,circle,#fff,#f4f7f9);background:radial-gradient(circle at 26% 19%,#fff,#f4f7f9);display:-webkit-box;display:-ms-flexbox;display:flex;-webkit-box-pack:center;-ms-flex-pack:center;justify-content:center;-webkit-box-align:center;-ms-flex-align:center;align-items:center;-ms-flex-line-pack:center;align-content:center;width:100%;min-height:100vh;line-height:1;flex-direction:column}p{display:block}main{text-align:center;flex:1;display:-webkit-box;display:-ms-flexbox;display:flex;-webkit-box-pack:center;-ms-flex-pack:center;justify-content:center;-webkit-box-align:center;-ms-flex-align:center;align-items:center;-ms-flex-line-pack:center;align-content:center;flex-direction:column}p{font-size:18px;line-height:normal;color:#646464;font-family:sans-serif;font-weight:400}a{color:#4842b7}footer{width:100%;text-align:center}footer p{font-size:16px}</style></head><body><main><p>Sorry, you cannot access this page. Please contact the customer service team.</p></main><footer><p>Security provided by <a href="https://www.datadoghq.com/product/security-platform/application-security-monitoring/" target="_blank">Datadog</a></p></footer></body></html>"""  # noqa: E501
BLOCKED_RESPONSE_JSON = '{"errors":[{"title":"You\'ve been blocked","detail":"Sorry, you cannot access this page. Please contact the customer service team. Security provided by Datadog."}]}'  # noqa: E501
HTTP_REQUEST_BLOCKED = "http.request.blocked"
RESPONSE_HEADERS = "http.response.headers"
HTTP_REQUEST_QUERY = "http.request.query"
HTTP_REQUEST_COOKIE_VALUE = "http.request.cookie.value"
HTTP_REQUEST_COOKIE_NAME = "http.request.cookie.name"
HTTP_REQUEST_PATH = "http.request.path"
HTTP_REQUEST_HEADER_NAME = "http.request.header.name"
HTTP_REQUEST_HEADER = "http.request.header"
HTTP_REQUEST_PARAMETER = "http.request.parameter"
HTTP_REQUEST_BODY = "http.request.body"
HTTP_REQUEST_PATH_PARAMETER = "http.request.path.parameter"
REQUEST_PATH_PARAMS = "http.request.path_params"
STATUS_403_TYPE_AUTO = {"status_code": 403, "type": "auto"}

CONTAINER_ID_HEADER_NAME = "Datadog-Container-Id"

ENTITY_ID_HEADER_NAME = "Datadog-Entity-ID"

EXTERNAL_ENV_HEADER_NAME = "Datadog-External-Env"
EXTERNAL_ENV_ENVIRONMENT_VARIABLE = "DD_EXTERNAL_ENV"

MESSAGING_SYSTEM = "messaging.system"

FLASK_ENDPOINT = "flask.endpoint"
FLASK_VIEW_ARGS = "flask.view_args"
FLASK_URL_RULE = "flask.url_rule"

_HTTPLIB_NO_TRACE_REQUEST = "_dd_no_trace"
DEFAULT_TIMEOUT = 2.0

# baggage
DD_TRACE_BAGGAGE_MAX_ITEMS = 64
DD_TRACE_BAGGAGE_MAX_BYTES = 8192


class _PRIORITY_CATEGORY:
    USER = "user"
    RULE_DEF = "rule_default"
    RULE_CUSTOMER = "rule_customer"
    RULE_DYNAMIC = "rule_dynamic"
    AUTO = "auto"
    DEFAULT = "default"


# intermediate mapping of priority categories to actual priority values
# used to simplify code that selects sampling priority based on many factors
_CATEGORY_TO_PRIORITIES = {
    _PRIORITY_CATEGORY.USER: (USER_KEEP, USER_REJECT),
    _PRIORITY_CATEGORY.RULE_DEF: (USER_KEEP, USER_REJECT),
    _PRIORITY_CATEGORY.RULE_CUSTOMER: (USER_KEEP, USER_REJECT),
    _PRIORITY_CATEGORY.RULE_DYNAMIC: (USER_KEEP, USER_REJECT),
    _PRIORITY_CATEGORY.AUTO: (AUTO_KEEP, AUTO_REJECT),
    _PRIORITY_CATEGORY.DEFAULT: (AUTO_KEEP, AUTO_REJECT),
}
_KEEP_PRIORITY_INDEX = 0
_REJECT_PRIORITY_INDEX = 1
