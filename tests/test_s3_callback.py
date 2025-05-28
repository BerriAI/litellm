"""
Test S3 callback logging with LiteLLM.

Requirements to run this test:
1. AWS credentials must be configured in environment variables:
   - AWS_ACCESS_KEY_ID: Your AWS access key
   - AWS_SECRET_ACCESS_KEY: Your AWS secret key
   - AWS_REGION_NAME: AWS region (default: us-east-1)

2. Anthropic API key must be set in environment variable:
   - ANTHROPIC_API_KEY: Your Anthropic API key

3. S3 bucket must exist with write permissions
   - Default bucket: "litellm-test-logs" (modify bucket_name variable if needed)

Usage:
    python tests/test_s3_callback.py
"""

import litellm
from litellm.integrations.s3 import S3Logger
import os
import uuid
from datetime import datetime

bucket_name = "litellm-test-logs"

access_key = os.environ.get('AWS_ACCESS_KEY_ID', 'not set')
secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY', 'not set')
if secret_key and len(secret_key) > 4:
    secret_preview = secret_key[:4] + '****' + secret_key[-4:]
else:
    secret_preview = 'invalid'

print(f"Using AWS credentials: Access Key: {access_key}, Secret Key: {secret_preview}")
print(f"Target S3 bucket: {bucket_name}")

test_id = str(uuid.uuid4())[:8]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
test_marker = f"test_{timestamp}_{test_id}"

litellm.s3_callback_params = {
    "s3_bucket_name": bucket_name,
    "s3_region_name": "us-east-1",
    "s3_path": f"tests/{test_marker}/"  # This is the correct parameter name, not s3_path_prefix
}

# Set the success_callback to tell LiteLLM to use the S3 logger for successful API calls
litellm.success_callback = ["s3"]

print(f"S3 callback params: {litellm.s3_callback_params}")
print(f"Success callbacks: {litellm.success_callback}")

try:
    s3_logger = S3Logger()
    litellm.callbacks = [s3_logger]
    print("S3 logger initialized successfully")
except Exception as e:
    print(f"Error initializing S3 logger: {e}")
    exit(1)

try:
    # Set the Anthropic API key from environment
    anthropic_api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")
    
    print("Making request to Claude...")
    response = litellm.completion(
        model="anthropic/claude-opus-4-20250514",
        api_key=anthropic_api_key,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Who is the president of the united states? (Test ID: {test_id})"},
        ],
        max_tokens=50,
    )
    
    print("\nResponse from Claude:")
    print(response)
    
    # Get today's date for S3 path
    today = datetime.now().strftime("%Y-%m-%d")
    
    print(f"\nS3 logging should be complete.")
    print(f"IMPORTANT: The S3 logger always includes a date-based folder structure.")
    print(f"Your logs should be at: s3://{bucket_name}/tests/{test_marker}/{today}/")
    print(f"But most likely they're just at: s3://{bucket_name}/{today}/")
    
    print(f"\nTROUBLESHOOTING TIPS:")
    print(f"1. Make sure both success_callback and callbacks are properly set:")
    print(f"   - litellm.success_callback = ['s3']")
    print(f"   - litellm.callbacks = [s3_logger]")
    print(f"2. Verify AWS credentials have write access to the S3 bucket")
    print(f"3. Check S3 path exists and is writable")
    
    print(f"\nCheck your bucket using:")
    print(f"aws s3 ls s3://{bucket_name}/{today}/")
    
    # Also suggest looking at the most recent file
    print(f"\nTo view the latest log file content:")
    print(f"aws s3 ls s3://{bucket_name}/{today}/ | sort | tail -n 1 | awk '{{print $4}}' | xargs -I{{}} aws s3 cp s3://{bucket_name}/{today}/{{}} -")
except Exception as e:
    print(f"\nError during completion: {e}")
    print("Make sure your ANTHROPIC_API_KEY environment variable is set correctly")