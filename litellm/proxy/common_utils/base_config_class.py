import asyncio
import os
from typing import TYPE_CHECKING, Any, Dict, Optional

import yaml

from litellm._logging import verbose_proxy_logger
from litellm.secret_managers.main import get_secret, get_secret_str

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient
else:
    PrismaClient = Any


class BaseProxyConfig:
    """
    Base class for proxy config.yaml

    Handles loading litellm config.yaml from a file, S3 bucket, GCS bucket or DB
    """

    def is_yaml(self, config_file_path: str) -> bool:
        if not os.path.isfile(config_file_path):
            return False

        _, file_extension = os.path.splitext(config_file_path)
        return file_extension.lower() == ".yaml" or file_extension.lower() == ".yml"

    async def get_config(
        self,
        config_file_path: Optional[str] = None,
    ) -> Dict:
        """
        Get the config.yaml file contents from a File, S3 bucket, GCS bucket or DB

        - Read from s3/GCS bucket when `LITELLM_CONFIG_BUCKET_NAME` is set in .env
        - Default to reading from config file path
        """
        # Load existing config
        if os.environ.get("LITELLM_CONFIG_BUCKET_NAME") is not None:
            bucket_name = os.environ.get("LITELLM_CONFIG_BUCKET_NAME")
            object_key = os.environ.get("LITELLM_CONFIG_BUCKET_OBJECT_KEY")
            bucket_type = os.environ.get("LITELLM_CONFIG_BUCKET_TYPE")
            verbose_proxy_logger.debug(
                "bucket_name: %s, object_key: %s", bucket_name, object_key
            )
            if bucket_type == "gcs":
                config = await self._get_config_file_contents_from_gcs(
                    bucket_name=bucket_name, object_key=object_key
                )
            else:
                config = self._get_file_contents_from_s3(
                    bucket_name=bucket_name, object_key=object_key
                )

            if config is None:
                raise Exception("Unable to load config from given source.")
        else:
            # default to file
            config = await self._get_config_from_file(config_file_path=config_file_path)

        config = self._check_for_os_environ_vars(config=config)
        return config

    async def _get_config_from_file(
        self, config_file_path: Optional[str] = None
    ) -> Dict:
        from litellm.proxy.proxy_server import (
            general_settings,
            prisma_client,
            store_model_in_db,
            user_config_file_path,
        )

        file_path = config_file_path or user_config_file_path
        if config_file_path is not None:
            user_config_file_path = config_file_path
        # Load existing config
        ## Yaml
        if os.path.exists(f"{file_path}"):
            with open(f"{file_path}", "r") as config_file:
                config = yaml.safe_load(config_file)
        elif config_file_path is not None:
            raise Exception(f"Config file not found at {config_file_path}")
        else:
            config = {
                "model_list": [],
                "general_settings": {},
                "router_settings": {},
                "litellm_settings": {},
            }

        ## DB
        if prisma_client is not None and (
            general_settings.get("store_model_in_db", False) is True
            or store_model_in_db is True
        ):
            config = await self._update_config_from_db(
                config=config, prisma_client=prisma_client
            )
        return config

    async def _update_config_from_db(
        self, config: Dict, prisma_client: PrismaClient
    ) -> Dict:
        _tasks = []
        keys = [
            "general_settings",
            "router_settings",
            "litellm_settings",
            "environment_variables",
        ]
        for k in keys:
            response = prisma_client.get_generic_data(
                key="param_name", value=k, table_name="config"
            )
            _tasks.append(response)

        responses = await asyncio.gather(*_tasks)
        for response in responses:
            if response is not None:
                param_name = getattr(response, "param_name", None)
                param_value = getattr(response, "param_value", None)
                if param_name is not None and param_value is not None:
                    # check if param_name is already in the config
                    if param_name in config:
                        if isinstance(config[param_name], dict):
                            config[param_name].update(param_value)
                        else:
                            config[param_name] = param_value
                    else:
                        # if it's not in the config - then add it
                        config[param_name] = param_value
        return config

    def _get_file_contents_from_s3(
        self, bucket_name: Optional[str] = None, object_key: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Get the config.yaml file contents from a S3 bucket
        """
        try:
            # v0 rely on boto3 for authentication - allowing boto3 to handle IAM credentials etc
            import tempfile

            import boto3
            from botocore.config import Config
            from botocore.credentials import Credentials

            from litellm.main import bedrock_converse_chat_completion

            credentials: Credentials = (
                bedrock_converse_chat_completion.get_credentials()
            )
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

            # Read the file contents
            file_contents = response["Body"].read().decode("utf-8")
            verbose_proxy_logger.debug("File contents retrieved from S3")

            # Create a temporary file with YAML extension
            with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml") as temp_file:
                temp_file.write(file_contents.encode("utf-8"))
                temp_file_path = temp_file.name
                verbose_proxy_logger.debug(
                    f"File stored temporarily at: {temp_file_path}"
                )

            # Load the YAML file content
            with open(temp_file_path, "r") as yaml_file:
                config = yaml.safe_load(yaml_file)

            return config
        except ImportError as e:
            # this is most likely if a user is not using the litellm docker container
            verbose_proxy_logger.error(f"ImportError: {str(e)}")
            return None
        except Exception as e:
            verbose_proxy_logger.error(f"Error retrieving file contents: {str(e)}")
            return None

    async def _get_config_file_contents_from_gcs(
        self, bucket_name: Optional[str] = None, object_key: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Get the config.yaml file contents from a GCS bucket
        """
        try:
            from litellm.integrations.gcs_bucket.gcs_bucket import GCSBucketLogger

            gcs_bucket = GCSBucketLogger(
                bucket_name=bucket_name,
            )
            if object_key is None:
                raise Exception(f"Object key is None for {bucket_name}")
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

    # # Example usage
    # bucket_name = 'litellm-proxy'
    # object_key = 'litellm_proxy_config.yaml'

    def _check_for_os_environ_vars(
        self, config: dict, depth: int = 0, max_depth: int = 10
    ) -> dict:
        """
        Check for os.environ/ variables in the config and replace them with the actual values.
        Includes a depth limit to prevent infinite recursion.

        Args:
            config (dict): The configuration dictionary to process.
            depth (int): Current recursion depth.
            max_depth (int): Maximum allowed recursion depth.

        Returns:
            dict: Processed configuration dictionary.
        """
        if depth > max_depth:
            verbose_proxy_logger.warning(
                f"Maximum recursion depth ({max_depth}) reached while processing config."
            )
            return config

        for key, value in config.items():
            if isinstance(value, dict):
                config[key] = self._check_for_os_environ_vars(
                    config=value, depth=depth + 1, max_depth=max_depth
                )
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        item = self._check_for_os_environ_vars(
                            config=item, depth=depth + 1, max_depth=max_depth
                        )
            # if the value is a string and starts with "os.environ/" - then it's an environment variable
            elif isinstance(value, str) and value.startswith("os.environ/"):
                config[key] = get_secret(value)
        return config

    async def save_config(self, new_config: dict):
        """
        Save the config.yaml contents to the DB (if user has opted in) or save in file
        """
        from litellm.proxy.proxy_server import (
            general_settings,
            prisma_client,
            store_model_in_db,
            user_config_file_path,
        )

        # Load existing config
        ## DB - writes valid config to db
        """
        - Do not write restricted params like 'api_key' to the database
        - if api_key is passed, save that to the local environment or connected secret manage (maybe expose `litellm.save_secret()`)
        """
        if prisma_client is not None and (
            general_settings.get("store_model_in_db", False) is True
            or store_model_in_db
        ):
            # if using - db for config - models are in ModelTable
            new_config.pop("model_list", None)
            await prisma_client.insert_data(data=new_config, table_name="config")
        else:
            # Save the updated config - if user is not using a dB
            ## YAML
            with open(f"{user_config_file_path}", "w") as config_file:
                yaml.dump(new_config, config_file, default_flow_style=False)
