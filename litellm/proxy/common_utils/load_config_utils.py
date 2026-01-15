import os

import yaml

from litellm._logging import verbose_proxy_logger


def get_file_contents_from_s3(bucket_name, object_key):
    try:
        # v0 rely on boto3 for authentication - allowing boto3 to handle IAM credentials etc
        import boto3
        from botocore.credentials import Credentials

        from litellm.main import bedrock_converse_chat_completion

        credentials: Credentials = bedrock_converse_chat_completion.get_credentials()
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=credentials.access_key,
            aws_secret_access_key=credentials.secret_key,
            aws_session_token=credentials.token,  # Optional, if using temporary credentials
        )
        verbose_proxy_logger.debug(
            f"Retrieving {object_key} from S3 bucket: {bucket_name}"
        )
        response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
        verbose_proxy_logger.debug(f"Response: {response}")

        # Read the file contents and directly parse YAML
        file_contents = response["Body"].read().decode("utf-8")
        verbose_proxy_logger.debug("File contents retrieved from S3")
        
        # Parse YAML directly from string
        config = yaml.safe_load(file_contents)
        return config

    except ImportError as e:
        # this is most likely if a user is not using the litellm docker container
        verbose_proxy_logger.error(f"ImportError: {str(e)}")
        pass
    except Exception as e:
        verbose_proxy_logger.error(f"Error retrieving file contents: {str(e)}")
        return None


async def get_config_file_contents_from_gcs(bucket_name, object_key):
    try:
        from litellm.integrations.gcs_bucket.gcs_bucket import GCSBucketLogger

        gcs_bucket = GCSBucketLogger(
            bucket_name=bucket_name,
        )
        file_contents = await gcs_bucket.download_gcs_object(object_key)
        if file_contents is None:
            raise Exception(f"File contents are None for {object_key}")
        # file_contentis is a bytes object, so we need to convert it to yaml
        file_contents = file_contents.decode("utf-8")
        # convert to yaml
        config = yaml.safe_load(file_contents)
        return config

    except Exception as e:
        verbose_proxy_logger.error(f"Error retrieving file contents: {str(e)}")
        return None


def download_python_file_from_s3(
    bucket_name: str,
    object_key: str,
    local_file_path: str,
) -> bool:
    """
    Download a Python file from S3 and save it to local filesystem.
    
    Args:
        bucket_name (str): S3 bucket name
        object_key (str): S3 object key (file path in bucket)
        local_file_path (str): Local path where file should be saved
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import boto3
        from botocore.credentials import Credentials

        from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
        base_aws_llm = BaseAWSLLM()

        credentials: Credentials = base_aws_llm.get_credentials()
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=credentials.access_key,
            aws_secret_access_key=credentials.secret_key,
            aws_session_token=credentials.token,
        )
        
        verbose_proxy_logger.debug(
            f"Downloading Python file {object_key} from S3 bucket: {bucket_name}"
        )
        response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
        
        # Read the file contents
        file_contents = response["Body"].read().decode("utf-8")
        verbose_proxy_logger.debug(f"File contents: {file_contents}")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        
        # Write to local file
        with open(local_file_path, 'w') as f:
            f.write(file_contents)
            
        verbose_proxy_logger.debug(f"Python file downloaded successfully to {local_file_path}")
        return True

    except ImportError as e:
        verbose_proxy_logger.error(f"ImportError: {str(e)}")
        return False
    except Exception as e:
        verbose_proxy_logger.exception(f"Error downloading Python file: {str(e)}")
        return False


async def download_python_file_from_gcs(
    bucket_name: str,
    object_key: str,
    local_file_path: str,
) -> bool:
    """
    Download a Python file from GCS and save it to local filesystem.
    
    Args:
        bucket_name (str): GCS bucket name
        object_key (str): GCS object key (file path in bucket)
        local_file_path (str): Local path where file should be saved
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        from litellm.integrations.gcs_bucket.gcs_bucket import GCSBucketLogger

        gcs_bucket = GCSBucketLogger(
            bucket_name=bucket_name,
        )
        file_contents = await gcs_bucket.download_gcs_object(object_key)
        if file_contents is None:
            raise Exception(f"File contents are None for {object_key}")
            
        # file_contents is a bytes object, decode it
        file_contents = file_contents.decode("utf-8")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        
        # Write to local file
        with open(local_file_path, 'w') as f:
            f.write(file_contents)
            
        verbose_proxy_logger.debug(f"Python file downloaded successfully to {local_file_path}")
        return True

    except Exception as e:
        verbose_proxy_logger.exception(f"Error downloading Python file from GCS: {str(e)}")
        return False


# # Example usage
# bucket_name = 'litellm-proxy'
# object_key = 'litellm_proxy_config.yaml'
