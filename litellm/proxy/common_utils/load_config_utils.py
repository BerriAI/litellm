from typing import Callable

import yaml

from litellm._logging import verbose_proxy_logger


async def get_file_contents_from_s3(bucket_name, object_key):
    try:
        # v0 rely on boto3 for authentication - allowing boto3 to handle IAM credentials etc
        import tempfile

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
        verbose_proxy_logger.error(
            f"Retrieving {object_key} from S3 bucket: {bucket_name}"
        )
        response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
        verbose_proxy_logger.error(f"Response: {response}")

        # Read the file contents
        file_contents = response["Body"].read().decode("utf-8")
        verbose_proxy_logger.error("File contents retrieved from S3")

        # Create a temporary file with YAML extension
        with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml") as temp_file:
            temp_file.write(file_contents.encode("utf-8"))
            temp_file_path = temp_file.name
            verbose_proxy_logger.error(f"File stored temporarily at: {temp_file_path}")

        # Load the YAML file content
        with open(temp_file_path, "r") as yaml_file:
            config = yaml.safe_load(yaml_file)

        # include file config
        config = await process_includes_from_bucket(
            config=config,
            get_file_method=get_file_contents_from_s3,
            bucket_name=bucket_name,
        )

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
        # include file config
        config = await process_includes_from_bucket(
            config=config,
            get_file_method=get_config_file_contents_from_gcs,
            bucket_name=bucket_name,
        )
        return config

    except Exception as e:
        verbose_proxy_logger.error(f"Error retrieving file contents: {str(e)}")
        return None


async def process_includes_from_bucket(
    config: dict, get_file_method: Callable, bucket_name: str
) -> dict:
    """
    Process includes by appending their contents to the main config

    Handles nested config.yamls with `include` section

    Example config: This will get the contents from files in `include` and append it
    ```yaml
    include:
        - /path/to/key/model_config.yaml

    litellm_settings:
        callbacks: ["prometheus"]
    ```
    """
    if "include" not in config:
        return config

    if not isinstance(config["include"], list):
        raise ValueError("'include' must be a list of file paths")

    # Load and append all included files
    for include_file in config["include"]:
        included_config = await get_file_method(bucket_name, include_file)
        # Simply update/extend the main config with included config
        for key, value in included_config.items():
            if isinstance(value, list) and key in config:
                config[key].extend(value)
            else:
                config[key] = value

    # Remove the include directive
    del config["include"]
    return config


# # Example usage
# bucket_name = 'litellm-proxy'
# object_key = 'litellm_proxy_config.yaml'
