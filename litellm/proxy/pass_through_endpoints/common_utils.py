from fastapi import Request


def get_litellm_virtual_key(request: Request) -> str:
    """
    Extract and format API key from request headers.
    Prioritizes x-litellm-api-key over Authorization header.


    Vertex JS SDK uses `Authorization` header, we use `x-litellm-api-key` to pass litellm virtual key

    """
    litellm_api_key = request.headers.get("x-litellm-api-key")
    if litellm_api_key:
        return f"Bearer {litellm_api_key}"
    return request.headers.get("Authorization", "")


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