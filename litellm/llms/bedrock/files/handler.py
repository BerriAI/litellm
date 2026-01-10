import asyncio
import base64
from typing import Any, Coroutine, Optional, Tuple, Union

import httpx

from litellm import LlmProviders
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.types.llms.openai import (
    FileContentRequest,
    HttpxBinaryResponseContent,
)
from litellm.types.utils import SpecialEnums

from ..base_aws_llm import BaseAWSLLM


class BedrockFilesHandler(BaseAWSLLM):
    """
    Handles downloading files from S3 for Bedrock batch processing.
    
    This implementation downloads files from S3 buckets where Bedrock
    stores batch output files.
    """

    def __init__(self):
        super().__init__()
        self.async_httpx_client = get_async_httpx_client(
            llm_provider=LlmProviders.BEDROCK,
        )

    def _extract_s3_uri_from_file_id(self, file_id: str) -> str:
        """
        Extract S3 URI from encoded file ID.
        
        The file ID can be in two formats:
        1. Base64-encoded unified file ID containing: llm_output_file_id,s3://bucket/path
        2. Direct S3 URI: s3://bucket/path
        
        Args:
            file_id: Encoded file ID or direct S3 URI
            
        Returns:
            S3 URI (e.g., "s3://bucket-name/path/to/file")
        """
        # First, try to decode if it's a base64-encoded unified file ID
        try:
            # Add padding if needed
            padded = file_id + "=" * (-len(file_id) % 4)
            decoded = base64.urlsafe_b64decode(padded).decode()
            
            # Check if it's a unified file ID format
            if decoded.startswith(SpecialEnums.LITELM_MANAGED_FILE_ID_PREFIX.value):
                # Extract llm_output_file_id from the decoded string
                if "llm_output_file_id," in decoded:
                    s3_uri = decoded.split("llm_output_file_id,")[1].split(";")[0]
                    return s3_uri
        except Exception:
            pass
        
        # If not base64 encoded or doesn't contain llm_output_file_id, assume it's already an S3 URI
        if file_id.startswith("s3://"):
            return file_id
        
        # If it doesn't start with s3://, assume it's a direct S3 URI and add the prefix
        return f"s3://{file_id}"

    def _parse_s3_uri(self, s3_uri: str) -> Tuple[str, str]:
        """
        Parse S3 URI to extract bucket name and object key.
        
        Args:
            s3_uri: S3 URI (e.g., "s3://bucket-name/path/to/file")
            
        Returns:
            Tuple of (bucket_name, object_key)
        """
        if not s3_uri.startswith("s3://"):
            raise ValueError(f"Invalid S3 URI format: {s3_uri}. Expected format: s3://bucket-name/path/to/file")
        
        # Remove 's3://' prefix
        path = s3_uri[5:]
        
        if "/" in path:
            bucket_name, object_key = path.split("/", 1)
        else:
            bucket_name = path
            object_key = ""
        
        return bucket_name, object_key

    def _find_bedrock_batch_output_file(self, s3_client, bucket_name: str, directory_prefix: str) -> str:
        """
        Find the actual Bedrock batch output file in the S3 directory structure.
        
        Bedrock batch jobs create output files in the format:
        {output_directory}/{model_invocation_id}/{input_file_name}.out
        
        Args:
            s3_client: Boto3 S3 client
            bucket_name: S3 bucket name
            directory_prefix: Directory prefix to search in
            
        Returns:
            Actual S3 object key for the output file
            
        Raises:
            ValueError: If no output file is found or multiple files are found
        """
        try:
            # List objects in the directory
            response = s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=directory_prefix,
                Delimiter='/'
            )
            
            # First, check if there are subdirectories (model invocation IDs)
            if 'CommonPrefixes' in response:
                # There are subdirectories, iterate through them to find output files
                output_files = []
                
                for prefix_info in response['CommonPrefixes']:
                    sub_prefix = prefix_info['Prefix']
                    
                    # List files in the subdirectory
                    sub_response = s3_client.list_objects_v2(
                        Bucket=bucket_name,
                        Prefix=sub_prefix
                    )
                    
                    if 'Contents' in sub_response:
                        for obj in sub_response['Contents']:
                            # Look for .out files (Bedrock batch output format)
                            if obj['Key'].endswith('.out'):
                                output_files.append(obj['Key'])
                
                if not output_files:
                    raise ValueError(f"No Bedrock batch output files (.out) found in s3://{bucket_name}/{directory_prefix}")
                
                if len(output_files) == 1:
                    return output_files[0]
                else:
                    # Multiple output files found - concatenate them into a single JSONL response
                    # This is common for batch jobs that split input files
                    return self._concatenate_bedrock_output_files(s3_client, bucket_name, output_files)
            
            # If no subdirectories, check for direct files in the directory
            elif 'Contents' in response:
                output_files = []
                for obj in response['Contents']:
                    # Skip the directory itself
                    if not obj['Key'].endswith('/'):
                        output_files.append(obj['Key'])
                
                if not output_files:
                    raise ValueError(f"No files found in s3://{bucket_name}/{directory_prefix}")
                
                if len(output_files) == 1:
                    return output_files[0]
                else:
                    # Multiple files found, prefer .out files
                    out_files = [f for f in output_files if f.endswith('.out')]
                    if out_files:
                        return out_files[0]
                    else:
                        return output_files[0]
            
            else:
                raise ValueError(f"No files or subdirectories found in s3://{bucket_name}/{directory_prefix}")
                
        except Exception as e:
            if "NoSuchKey" in str(e) or "does not exist" in str(e):
                raise ValueError(f"Directory s3://{bucket_name}/{directory_prefix} does not exist or is empty. This may indicate that the Bedrock batch job has not completed yet or failed.")
            else:
                raise ValueError(f"Error searching for Bedrock batch output files in s3://{bucket_name}/{directory_prefix}. Error: {str(e)}")

    def _concatenate_bedrock_output_files(self, s3_client, bucket_name: str, output_file_keys: list) -> str:
        """
        Concatenate multiple Bedrock batch output files into a single JSONL content.
        
        Args:
            s3_client: Boto3 S3 client
            bucket_name: S3 bucket name
            output_file_keys: List of S3 object keys for output files
            
        Returns:
            A special object key indicating concatenated content (will be handled differently in download)
        """
        # Return a special marker that indicates we need to concatenate files
        # The actual concatenation will happen in the download logic
        return f"__CONCATENATE__:{','.join(output_file_keys)}"

    async def afile_content(
        self,
        file_content_request: FileContentRequest,
        optional_params: dict,
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
    ) -> HttpxBinaryResponseContent:
        """
        Download file content from S3 bucket for Bedrock files.
        
        Args:
            file_content_request: Contains file_id (encoded or S3 URI)
            optional_params: Optional parameters containing AWS credentials
            timeout: Request timeout
            max_retries: Max retry attempts
            
        Returns:
            HttpxBinaryResponseContent: Binary content wrapped in compatible response format
        """
        import boto3
        from botocore.credentials import Credentials
        
        file_id = file_content_request.get("file_id")
        if not file_id:
            raise ValueError("file_id is required in file_content_request")
        
        # Extract S3 URI from file ID
        s3_uri = self._extract_s3_uri_from_file_id(file_id)
        bucket_name, object_key = self._parse_s3_uri(s3_uri)
        
        # Get AWS credentials
        aws_region_name = self._get_aws_region_name(
            optional_params=optional_params, model=""
        )
        credentials: Credentials = self.get_credentials(
            aws_access_key_id=optional_params.get("aws_access_key_id"),
            aws_secret_access_key=optional_params.get("aws_secret_access_key"),
            aws_session_token=optional_params.get("aws_session_token"),
            aws_region_name=aws_region_name,
            aws_session_name=optional_params.get("aws_session_name"),
            aws_profile_name=optional_params.get("aws_profile_name"),
            aws_role_name=optional_params.get("aws_role_name"),
            aws_web_identity_token=optional_params.get("aws_web_identity_token"),
            aws_sts_endpoint=optional_params.get("aws_sts_endpoint"),
        )
        
        # Create S3 client
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=credentials.access_key,
            aws_secret_access_key=credentials.secret_key,
            aws_session_token=credentials.token,
            region_name=aws_region_name,
            verify=self._get_ssl_verify(),
        )
        
        # Check if the object_key is a directory (ends with '/') or doesn't exist as a direct file
        # This handles the case where Bedrock batch outputs are stored in subdirectories
        actual_object_key = object_key
        
        if object_key.endswith('/') or not object_key:
            # This is a directory path, we need to find the actual output file(s)
            # Bedrock stores batch outputs as: {output_directory}/{model_invocation_id}/{input_file_name}.out
            actual_object_key = self._find_bedrock_batch_output_file(s3_client, bucket_name, object_key)
        else:
            # Try to download the file directly first
            try:
                response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
                file_content = response["Body"].read()
                
                # Create mock HTTP response
                mock_response = httpx.Response(
                    status_code=200,
                    content=file_content,
                    headers={"content-type": "application/octet-stream"},
                    request=httpx.Request(method="GET", url=s3_uri),
                )
                
                return HttpxBinaryResponseContent(response=mock_response)
            except Exception:
                # If direct download fails, try to find the file in subdirectories
                # This handles cases where the S3 URI doesn't end with '/' but still points to a directory
                actual_object_key = self._find_bedrock_batch_output_file(s3_client, bucket_name, object_key + "/")
        
        # Download file from S3 using the actual object key
        try:
            # Check if we need to concatenate multiple files
            if actual_object_key.startswith("__CONCATENATE__:"):
                file_keys = actual_object_key.split(":", 1)[1].split(",")
                concatenated_content = []
                
                for file_key in file_keys:
                    response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
                    content = response["Body"].read().decode('utf-8')
                    # Add each file's content, ensuring proper JSONL formatting
                    content = content.strip()
                    if content:
                        concatenated_content.append(content)
                
                # Join all content with newlines for proper JSONL format
                file_content = '\n'.join(concatenated_content).encode('utf-8')
                final_s3_uri = f"s3://{bucket_name}/{object_key}[concatenated-{len(file_keys)}-files]"
            else:
                response = s3_client.get_object(Bucket=bucket_name, Key=actual_object_key)
                file_content = response["Body"].read()
                final_s3_uri = f"s3://{bucket_name}/{actual_object_key}"
        except Exception as e:
            raise ValueError(f"Failed to download file from S3: s3://{bucket_name}/{actual_object_key}. Error: {str(e)}")
        
        # Create mock HTTP response
        mock_response = httpx.Response(
            status_code=200,
            content=file_content,
            headers={"content-type": "application/octet-stream"},
            request=httpx.Request(method="GET", url=final_s3_uri),
        )
        
        return HttpxBinaryResponseContent(response=mock_response)

    def file_content(
        self,
        _is_async: bool,
        file_content_request: FileContentRequest,
        api_base: Optional[str],
        optional_params: dict,
        timeout: Union[float, httpx.Timeout],
        max_retries: Optional[int],
    ) -> Union[
        HttpxBinaryResponseContent, Coroutine[Any, Any, HttpxBinaryResponseContent]
    ]:
        """
        Download file content from S3 bucket for Bedrock files.
        Supports both sync and async operations.
        
        Args:
            _is_async: Whether to run asynchronously
            file_content_request: Contains file_id (encoded or S3 URI)
            api_base: API base (unused for S3 operations)
            optional_params: Optional parameters containing AWS credentials
            timeout: Request timeout
            max_retries: Max retry attempts
            
        Returns:
            HttpxBinaryResponseContent or Coroutine: Binary content wrapped in compatible response format
        """
        if _is_async:
            return self.afile_content(
                file_content_request=file_content_request,
                optional_params=optional_params,
                timeout=timeout,
                max_retries=max_retries,
            )
        else:
            return asyncio.run(
                self.afile_content(
                    file_content_request=file_content_request,
                    optional_params=optional_params,
                    timeout=timeout,
                    max_retries=max_retries,
                )
            )

