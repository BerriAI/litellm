from typing import Dict, List

from mangum.handlers.utils import (
    handle_base64_response_body,
    handle_exclude_headers,
    handle_multi_value_headers,
    maybe_encode_body,
)
from mangum.types import Scope, Response, LambdaConfig, LambdaEvent, LambdaContext


class LambdaAtEdge:
    @classmethod
    def infer(
        cls, event: LambdaEvent, context: LambdaContext, config: LambdaConfig
    ) -> bool:
        return (
            "Records" in event
            and len(event["Records"]) > 0
            and "cf" in event["Records"][0]
        )

        # FIXME: Since this is the last in the chain it doesn't get coverage by default,
        # # just ignoring it for now.
        # return None  # pragma: nocover

    def __init__(
        self, event: LambdaEvent, context: LambdaContext, config: LambdaConfig
    ) -> None:
        self.event = event
        self.context = context
        self.config = config

    @property
    def body(self) -> bytes:
        cf_request_body = self.event["Records"][0]["cf"]["request"].get("body", {})
        return maybe_encode_body(
            cf_request_body.get("data"),
            is_base64=cf_request_body.get("encoding", "") == "base64",
        )

    @property
    def scope(self) -> Scope:
        cf_request = self.event["Records"][0]["cf"]["request"]
        scheme_header = cf_request["headers"].get("cloudfront-forwarded-proto", [{}])
        scheme = scheme_header[0].get("value", "https")
        host_header = cf_request["headers"].get("host", [{}])
        server_name = host_header[0].get("value", "mangum")
        if ":" not in server_name:
            forwarded_port_header = cf_request["headers"].get("x-forwarded-port", [{}])
            server_port = forwarded_port_header[0].get("value", 80)
        else:
            server_name, server_port = server_name.split(":")  # pragma: no cover

        server = (server_name, int(server_port))
        source_ip = cf_request["clientIp"]
        client = (source_ip, 0)
        http_method = cf_request["method"]

        return {
            "type": "http",
            "method": http_method,
            "http_version": "1.1",
            "headers": [
                [k.encode(), v[0]["value"].encode()]
                for k, v in cf_request["headers"].items()
            ],
            "path": cf_request["uri"],
            "raw_path": None,
            "root_path": "",
            "scheme": scheme,
            "query_string": cf_request["querystring"].encode(),
            "server": server,
            "client": client,
            "asgi": {"version": "3.0", "spec_version": "2.0"},
            "aws.event": self.event,
            "aws.context": self.context,
        }

    def __call__(self, response: Response) -> dict:
        multi_value_headers, _ = handle_multi_value_headers(response["headers"])
        response_body, is_base64_encoded = handle_base64_response_body(
            response["body"], multi_value_headers, self.config["text_mime_types"]
        )
        finalized_headers: Dict[str, List[Dict[str, str]]] = {
            key.decode().lower(): [{"key": key.decode().lower(), "value": val.decode()}]
            for key, val in response["headers"]
        }

        return {
            "status": response["status"],
            "headers": handle_exclude_headers(finalized_headers, self.config),
            "body": response_body,
            "isBase64Encoded": is_base64_encoded,
        }
