"""
Enterprise specific logging utils
"""
from litellm.litellm_core_utils.litellm_logging import StandardLoggingMetadata


class StandardLoggingPayloadSetup:
    @staticmethod
    def apply_enterprise_specific_metadata(
        standard_logging_metadata: StandardLoggingMetadata,
        proxy_server_request: dict,
    ) -> StandardLoggingMetadata:
        """
        Adds enterprise-only metadata to the standard logging metadata.
        """

        _request_headers = proxy_server_request.get("headers", {})

        if _request_headers:
            custom_headers = {
                k: v
                for k, v in _request_headers.items()
                if k.startswith("x-") and v is not None and isinstance(v, str)
            }

            standard_logging_metadata["requester_custom_headers"] = custom_headers

        return standard_logging_metadata
