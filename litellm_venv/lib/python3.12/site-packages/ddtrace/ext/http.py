"""
Standard http tags.

For example:

span.set_tag(URL, '/user/home')
span.set_tag(STATUS_CODE, 404)
"""
# tags
URL = "http.url"
METHOD = "http.method"
STATUS_CODE = "http.status_code"
USER_AGENT = "http.useragent"
STATUS_MSG = "http.status_msg"
QUERY_STRING = "http.query.string"
RETRIES_REMAIN = "http.retries_remain"
VERSION = "http.version"
CLIENT_IP = "http.client_ip"
ROUTE = "http.route"

# template render span type
TEMPLATE = "template"
