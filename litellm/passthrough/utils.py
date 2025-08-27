from typing import Dict, List, Optional, Union
from urllib.parse import parse_qs

import httpx


class BasePassthroughUtils:
    @staticmethod
    def get_merged_query_parameters(
        existing_url: httpx.URL, request_query_params: Dict[str, Union[str, list]]
    ) -> Dict[str, Union[str, List[str]]]:
        # Get the existing query params from the target URL
        existing_query_string = existing_url.query.decode("utf-8")
        existing_query_params = parse_qs(existing_query_string)

        # parse_qs returns a dict where each value is a list, so let's flatten it
        updated_existing_query_params = {
            k: v[0] if len(v) == 1 else v for k, v in existing_query_params.items()
        }
        # Merge the query params, giving priority to the existing ones
        return {**request_query_params, **updated_existing_query_params}

    @staticmethod
    def forward_headers_from_request(
        request_headers: dict,
        headers: dict,
        forward_headers: Optional[bool] = False,
    ):
        """
        Helper to forward headers from original request
        """
        if forward_headers is True:
            # Header We Should NOT forward
            request_headers.pop("content-length", None)
            request_headers.pop("host", None)

            # Combine request headers with custom headers
            headers = {**request_headers, **headers}
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
        if 'arn:aws:' not in endpoint:
            return endpoint

        # Handle all patterns in one go - more efficient and cleaner
        patterns = [
            # Custom model with 2 slashes (order matters - do this first)
            (r'(custom-model)/([a-z0-9.-]+)/([a-z0-9]+)', r'\1%2F\2%2F\3'),

            # All other resource types with 1 slash
            (r'(:application-inference-profile)/', r'\1%2F'),
            (r'(:inference-profile)/', r'\1%2F'),
            (r'(:foundation-model)/', r'\1%2F'),
            (r'(:imported-model)/', r'\1%2F'),
            (r'(:provisioned-model)/', r'\1%2F'),
            (r'(:prompt)/', r'\1%2F'),
            (r'(:endpoint)/', r'\1%2F'),
            (r'(:prompt-router)/', r'\1%2F'),
            (r'(:default-prompt-router)/', r'\1%2F'),
        ]

        for pattern, replacement in patterns:
            # Check if pattern exists before applying regex (early exit optimization)
            if re.search(pattern, endpoint):
                endpoint = re.sub(pattern, replacement, endpoint)
                break  # Exit after first match since each ARN has only one resource type

        return endpoint