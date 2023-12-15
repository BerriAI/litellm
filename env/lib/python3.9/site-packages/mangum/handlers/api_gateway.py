from typing import Dict, List, Tuple
from urllib.parse import urlencode

from mangum.handlers.utils import (
    get_server_and_port,
    handle_base64_response_body,
    handle_exclude_headers,
    handle_multi_value_headers,
    maybe_encode_body,
    strip_api_gateway_path,
)
from mangum.types import (
    Response,
    LambdaConfig,
    Headers,
    LambdaEvent,
    LambdaContext,
    QueryParams,
    Scope,
)


def _encode_query_string_for_apigw(event: LambdaEvent) -> bytes:
    params: QueryParams = event.get("multiValueQueryStringParameters", {})
    if not params:
        params = event.get("queryStringParameters", {})
    if not params:
        return b""

    return urlencode(params, doseq=True).encode()


def _handle_multi_value_headers_for_request(event: LambdaEvent) -> Dict[str, str]:
    headers = event.get("headers", {}) or {}
    headers = {k.lower(): v for k, v in headers.items()}
    if event.get("multiValueHeaders"):
        headers.update(
            {
                k.lower(): ", ".join(v) if isinstance(v, list) else ""
                for k, v in event.get("multiValueHeaders", {}).items()
            }
        )

    return headers


def _combine_headers_v2(
    input_headers: Headers,
) -> Tuple[Dict[str, str], List[str]]:
    output_headers: Dict[str, str] = {}
    cookies: List[str] = []
    for key, value in input_headers:
        normalized_key: str = key.decode().lower()
        normalized_value: str = value.decode()
        if normalized_key == "set-cookie":
            cookies.append(normalized_value)
        else:
            if normalized_key in output_headers:
                normalized_value = (
                    f"{output_headers[normalized_key]},{normalized_value}"
                )
            output_headers[normalized_key] = normalized_value

    return output_headers, cookies


class APIGateway:
    @classmethod
    def infer(
        cls, event: LambdaEvent, context: LambdaContext, config: LambdaConfig
    ) -> bool:
        return "resource" in event and "requestContext" in event

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
        headers = _handle_multi_value_headers_for_request(self.event)
        return {
            "type": "http",
            "http_version": "1.1",
            "method": self.event["httpMethod"],
            "headers": [[k.encode(), v.encode()] for k, v in headers.items()],
            "path": strip_api_gateway_path(
                self.event["path"],
                api_gateway_base_path=self.config["api_gateway_base_path"],
            ),
            "raw_path": None,
            "root_path": "",
            "scheme": headers.get("x-forwarded-proto", "https"),
            "query_string": _encode_query_string_for_apigw(self.event),
            "server": get_server_and_port(headers),
            "client": (
                self.event["requestContext"].get("identity", {}).get("sourceIp"),
                0,
            ),
            "asgi": {"version": "3.0", "spec_version": "2.0"},
            "aws.event": self.event,
            "aws.context": self.context,
        }

    def __call__(self, response: Response) -> dict:
        finalized_headers, multi_value_headers = handle_multi_value_headers(
            response["headers"]
        )
        finalized_body, is_base64_encoded = handle_base64_response_body(
            response["body"], finalized_headers, self.config["text_mime_types"]
        )

        return {
            "statusCode": response["status"],
            "headers": handle_exclude_headers(finalized_headers, self.config),
            "multiValueHeaders": handle_exclude_headers(
                multi_value_headers, self.config
            ),
            "body": finalized_body,
            "isBase64Encoded": is_base64_encoded,
        }


class HTTPGateway:
    @classmethod
    def infer(
        cls, event: LambdaEvent, context: LambdaContext, config: LambdaConfig
    ) -> bool:
        return "version" in event and "requestContext" in event

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
        request_context = self.event["requestContext"]
        event_version = self.event["version"]

        # API Gateway v2
        if event_version == "2.0":
            headers = {k.lower(): v for k, v in self.event.get("headers", {}).items()}
            source_ip = request_context["http"]["sourceIp"]
            path = request_context["http"]["path"]
            http_method = request_context["http"]["method"]
            query_string = self.event.get("rawQueryString", "").encode()

            if self.event.get("cookies"):
                headers["cookie"] = "; ".join(self.event.get("cookies", []))

        # API Gateway v1
        else:
            headers = _handle_multi_value_headers_for_request(self.event)
            source_ip = request_context.get("identity", {}).get("sourceIp")
            path = self.event["path"]
            http_method = self.event["httpMethod"]
            query_string = _encode_query_string_for_apigw(self.event)

        path = strip_api_gateway_path(
            path,
            api_gateway_base_path=self.config["api_gateway_base_path"],
        )
        server = get_server_and_port(headers)
        client = (source_ip, 0)

        return {
            "type": "http",
            "method": http_method,
            "http_version": "1.1",
            "headers": [[k.encode(), v.encode()] for k, v in headers.items()],
            "path": path,
            "raw_path": None,
            "root_path": "",
            "scheme": headers.get("x-forwarded-proto", "https"),
            "query_string": query_string,
            "server": server,
            "client": client,
            "asgi": {"version": "3.0", "spec_version": "2.0"},
            "aws.event": self.event,
            "aws.context": self.context,
        }

    def __call__(self, response: Response) -> dict:
        if self.scope["aws.event"]["version"] == "2.0":
            finalized_headers, cookies = _combine_headers_v2(response["headers"])

            if "content-type" not in finalized_headers and response["body"] is not None:
                finalized_headers["content-type"] = "application/json"

            finalized_body, is_base64_encoded = handle_base64_response_body(
                response["body"], finalized_headers, self.config["text_mime_types"]
            )
            response_out = {
                "statusCode": response["status"],
                "body": finalized_body,
                "headers": finalized_headers or None,
                "cookies": cookies or None,
                "isBase64Encoded": is_base64_encoded,
            }
            return {
                key: value for key, value in response_out.items() if value is not None
            }

        finalized_headers, multi_value_headers = handle_multi_value_headers(
            response["headers"]
        )
        finalized_body, is_base64_encoded = handle_base64_response_body(
            response["body"], finalized_headers, self.config["text_mime_types"]
        )
        return {
            "statusCode": response["status"],
            "headers": finalized_headers,
            "multiValueHeaders": multi_value_headers,
            "body": finalized_body,
            "isBase64Encoded": is_base64_encoded,
        }
