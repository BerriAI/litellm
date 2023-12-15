from itertools import islice
from typing import Dict, Generator, List, Tuple
from urllib.parse import urlencode, unquote, unquote_plus

from mangum.handlers.utils import (
    get_server_and_port,
    handle_base64_response_body,
    handle_exclude_headers,
    maybe_encode_body,
)
from mangum.types import (
    Response,
    Scope,
    LambdaConfig,
    LambdaEvent,
    LambdaContext,
    QueryParams,
)


def all_casings(input_string: str) -> Generator[str, None, None]:
    """
    Permute all casings of a given string.
    A pretty algoritm, via @Amber
    http://stackoverflow.com/questions/6792803/finding-all-possible-case-permutations-in-python
    """
    if not input_string:
        yield ""
    else:
        first = input_string[:1]
        if first.lower() == first.upper():
            for sub_casing in all_casings(input_string[1:]):
                yield first + sub_casing
        else:
            for sub_casing in all_casings(input_string[1:]):
                yield first.lower() + sub_casing
                yield first.upper() + sub_casing


def case_mutated_headers(multi_value_headers: Dict[str, List[str]]) -> Dict[str, str]:
    """Create str/str key/value headers, with duplicate keys case mutated."""
    headers: Dict[str, str] = {}
    for key, values in multi_value_headers.items():
        if len(values) > 0:
            casings = list(islice(all_casings(key), len(values)))
            for value, cased_key in zip(values, casings):
                headers[cased_key] = value
    return headers


def encode_query_string_for_alb(params: QueryParams) -> bytes:
    """Encode the query string parameters for the ALB event. The parameters must be
    decoded and then encoded again to prevent double encoding.

    According to the docs:

        "If the query parameters are URL-encoded, the load balancer does not decode
        "them. You must decode them in your Lambda function."
    """
    params = {
        unquote_plus(key): unquote_plus(value)
        if isinstance(value, str)
        else tuple(unquote_plus(element) for element in value)
        for key, value in params.items()
    }
    query_string = urlencode(params, doseq=True).encode()

    return query_string


def transform_headers(event: LambdaEvent) -> List[Tuple[bytes, bytes]]:
    headers: List[Tuple[bytes, bytes]] = []
    if "multiValueHeaders" in event:
        for k, v in event["multiValueHeaders"].items():
            for inner_v in v:
                headers.append((k.lower().encode(), inner_v.encode()))
    else:
        for k, v in event["headers"].items():
            headers.append((k.lower().encode(), v.encode()))

    return headers


class ALB:
    @classmethod
    def infer(
        cls, event: LambdaEvent, context: LambdaContext, config: LambdaConfig
    ) -> bool:
        return "requestContext" in event and "elb" in event["requestContext"]

    def __init__(
        self, event: LambdaEvent, context: LambdaContext, config: LambdaConfig
    ) -> None:
        self.event = event
        self.context = context
        self.config = config

    @property
    def body(self) -> bytes:
        return maybe_encode_body(
            self.event.get("body", b""),
            is_base64=self.event.get("isBase64Encoded", False),
        )

    @property
    def scope(self) -> Scope:

        headers = transform_headers(self.event)
        list_headers = [list(x) for x in headers]
        # Unique headers. If there are duplicates, it will use the last defined.
        uq_headers = {k.decode(): v.decode() for k, v in headers}
        source_ip = uq_headers.get("x-forwarded-for", "")
        path = unquote(self.event["path"]) if self.event["path"] else "/"
        http_method = self.event["httpMethod"]

        params = self.event.get(
            "multiValueQueryStringParameters",
            self.event.get("queryStringParameters", {}),
        )
        if not params:
            query_string = b""
        else:
            query_string = encode_query_string_for_alb(params)

        server = get_server_and_port(uq_headers)
        client = (source_ip, 0)

        scope: Scope = {
            "type": "http",
            "method": http_method,
            "http_version": "1.1",
            "headers": list_headers,
            "path": path,
            "raw_path": None,
            "root_path": "",
            "scheme": uq_headers.get("x-forwarded-proto", "https"),
            "query_string": query_string,
            "server": server,
            "client": client,
            "asgi": {"version": "3.0", "spec_version": "2.0"},
            "aws.event": self.event,
            "aws.context": self.context,
        }

        return scope

    def __call__(self, response: Response) -> dict:
        multi_value_headers: Dict[str, List[str]] = {}
        for key, value in response["headers"]:
            lower_key = key.decode().lower()
            if lower_key not in multi_value_headers:
                multi_value_headers[lower_key] = []
            multi_value_headers[lower_key].append(value.decode())

        finalized_headers = case_mutated_headers(multi_value_headers)
        finalized_body, is_base64_encoded = handle_base64_response_body(
            response["body"], finalized_headers, self.config["text_mime_types"]
        )

        out = {
            "statusCode": response["status"],
            "body": finalized_body,
            "isBase64Encoded": is_base64_encoded,
        }

        # You must use multiValueHeaders if you have enabled multi-value headers and
        # headers otherwise.
        multi_value_headers_enabled = "multiValueHeaders" in self.scope["aws.event"]
        if multi_value_headers_enabled:
            out["multiValueHeaders"] = handle_exclude_headers(
                multi_value_headers, self.config
            )
        else:
            out["headers"] = handle_exclude_headers(finalized_headers, self.config)

        return out
