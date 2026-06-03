from typing import Dict, List, Mapping, Optional, Union
from urllib.parse import parse_qs

import httpx

from litellm._logging import verbose_logger
from litellm.constants import PASS_THROUGH_HEADER_PREFIX

# Headers that must not be overwritten via the x-pass- forwarding mechanism.
# Includes standard credential/auth headers and protocol-level headers that
# affect routing or message framing.
_PASS_THROUGH_PROTECTED_HEADERS: frozenset = frozenset(
    {
        "authorization",
        "api-key",
        "x-api-key",
        "x-goog-api-key",
        "host",
        "content-length",
    }
)

# Header name prefix used to block AWS SigV4 signing headers from being overridden.
_PASS_THROUGH_PROTECTED_HEADER_PREFIXES: tuple = ("x-amz-",)


class BasePassthroughUtils:
    @staticmethod
    def get_merged_query_parameters(
        existing_url: httpx.URL,
        request_query_params: Mapping[str, Union[str, list]],
        default_query_params: Optional[Dict[str, Union[str, list]]] = None,
    ) -> Dict[str, Union[str, List[str]]]:
        # Get the existing query params from the target URL
        existing_query_string = existing_url.query.decode("utf-8")
        existing_query_params = parse_qs(existing_query_string)

        # parse_qs returns a dict where each value is a list, so let's flatten it
        updated_existing_query_params = {
            k: v[0] if len(v) == 1 else v for k, v in existing_query_params.items()
        }

        # Start with default query params (lowest priority)
        merged_params = {}
        if default_query_params:
            merged_params.update(default_query_params)

        # Override with existing URL query params (medium priority)
        merged_params.update(updated_existing_query_params)

        # Override with request query params (highest priority - client can override anything)
        merged_params.update(request_query_params)

        return merged_params

    @staticmethod
    def forward_headers_from_request(
        request_headers: dict,
        headers: dict,
        forward_headers: Optional[bool] = False,
    ):
        """
        Helper to forward headers from original request.

        Also handles 'x-pass-' prefixed headers which are always forwarded
        with the prefix stripped, regardless of forward_headers setting.
        e.g., 'x-pass-anthropic-beta: value' becomes 'anthropic-beta: value'

        Security (LIT-3550): when ``headers`` (operator-controlled, e.g.
        an upstream provider credential set by the per-provider passthrough
        route) contains any credential header, ALL inbound credential
        headers are dropped before merging. Inbound credential headers
        carry the LiteLLM virtual key, not the upstream provider
        credential, and must never reach the provider -- otherwise the
        upstream rejects the request as an invalid token (the underlying
        LIT-3550 / Anthropic passthrough symptom).
        """
        if forward_headers is True:
            # Header We Should NOT forward
            request_headers.pop("content-length", None)
            request_headers.pop("host", None)

            # HTTP header names are case-insensitive (RFC 9110); compare
            # in lowercase to avoid duplicate-but-different-case header
            # pairs in the merged dict.
            custom_header_names_lower = {h.lower() for h in headers}

            # Treat any credential header set on ``headers`` as the
            # authoritative upstream auth and drop inbound credential
            # headers of *any* type so the LiteLLM virtual key cannot
            # leak upstream.
            _credential_headers = {
                "authorization",
                "api-key",
                "x-api-key",
                "x-goog-api-key",
            }
            operator_set_credential = bool(
                custom_header_names_lower & _credential_headers
            )

            for header_name in list(request_headers.keys()):
                lname = header_name.lower()
                if lname in custom_header_names_lower:
                    # Operator-set header wins; drop the inbound to
                    # avoid case-mismatched duplicates at the HTTP layer.
                    request_headers.pop(header_name, None)
                elif operator_set_credential and lname in _credential_headers:
                    # Defense in depth: never forward the inbound LiteLLM
                    # virtual key as an upstream credential.
                    request_headers.pop(header_name, None)

            # Combine request headers with custom headers
            headers = {**request_headers, **headers}

        # Process x-pass- prefixed headers (strip prefix and forward)
        # Credential and protocol-level headers are excluded from this mechanism.
        for header_name, header_value in request_headers.items():
            if header_name.lower().startswith(PASS_THROUGH_HEADER_PREFIX):
                # Strip the 'x-pass-' prefix and normalize to lowercase
                actual_header_name = header_name[
                    len(PASS_THROUGH_HEADER_PREFIX) :
                ].lower()
                if actual_header_name in _PASS_THROUGH_PROTECTED_HEADERS or any(
                    actual_header_name.startswith(p)
                    for p in _PASS_THROUGH_PROTECTED_HEADER_PREFIXES
                ):
                    verbose_logger.debug(
                        "x-pass- header %s maps to a protected header name; skipping",
                        header_name,
                    )
                    continue
                headers[actual_header_name] = header_value

        return headers


class CommonUtils:
    @staticmethod
    def encode_bedrock_runtime_modelid_arn(endpoint: str) -> str:
        """
        Encodes any "/" found in the modelId of an AWS Bedrock Runtime Endpoint when arns are passed in.
        - modelID value can be an ARN which contains slashes that SHOULD NOT be treated as path separators.
        e.g endpoint: /model/<modelId>/invoke
        <modelId> containing arns with slashes need to be encoded from
            arn:aws:bedrock:ap-southeast-1:123456789012:application-inference-profile/abdefg12334 =>
            arn:aws:bedrock:ap-southeast-1:123456789012:application-inference-profile%2Fabdefg12334
        so that it is treated as one part of the path.
        Otherwise, the encoded endpoint will return 500 error when passed to Bedrock endpoint.

        See the apis in https://docs.aws.amazon.com/bedrock/latest/APIReference/API_Operations_Amazon_Bedrock_Runtime.html
        for more details on the regex patterns of modelId which we use in the regex logic below.

        Args:
            endpoint (str): The original endpoint string which may contain ARNs that contain slashes.

        Returns:
            str: The endpoint with properly encoded ARN slashes
        """
        import re

        # Early exit: if no ARN detected, return unchanged
        if "arn:aws:" not in endpoint:
            return endpoint

        # Handle all patterns in one go - more efficient and cleaner
        patterns = [
            # Custom model with 2 slashes (order matters - do this first)
            (r"(custom-model)/([a-z0-9.-]+)/([a-z0-9]+)", r"\1%2F\2%2F\3"),
            # All other resource types with 1 slash
            (r"(:application-inference-profile)/", r"\1%2F"),
            (r"(:inference-profile)/", r"\1%2F"),
            (r"(:foundation-model)/", r"\1%2F"),
            (r"(:imported-model)/", r"\1%2F"),
            (r"(:provisioned-model)/", r"\1%2F"),
            (r"(:prompt)/", r"\1%2F"),
            (r"(:endpoint)/", r"\1%2F"),
            (r"(:prompt-router)/", r"\1%2F"),
            (r"(:default-prompt-router)/", r"\1%2F"),
        ]

        for pattern, replacement in patterns:
            # Check if pattern exists before applying regex (early exit optimization)
            if re.search(pattern, endpoint):
                endpoint = re.sub(pattern, replacement, endpoint)
                break  # Exit after first match since each ARN has only one resource type

        return endpoint
