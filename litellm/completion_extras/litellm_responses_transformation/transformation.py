"""
Handler for transforming /chat/completions api requests to litellm.responses requests
"""


class LiteLLMResponsesTransformationHandler:
    """
    Handler for transforming /chat/completions api requests to litellm.responses requests
    """

    def __init__(self):
        pass

    def transform_request(self, request: dict) -> dict:
        """Transform /chat/completions api request to litellm.responses request"""
        return request

    def transform_response(self, response: dict) -> dict:
        """Transform litellm.responses response to /chat/completions api response"""
        return response

    def get_sync_custom_stream_wrapper(self, request: dict) -> dict:
        """Get sync custom stream wrapper for litellm.responses request"""
        return request

    def get_async_custom_stream_wrapper(self, request: dict) -> dict:
        """Get async custom stream wrapper for litellm.responses request"""
        return request
