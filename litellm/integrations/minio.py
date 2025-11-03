"""
MinIO Logging Integration

Logs events to MinIO (S3-compatible object storage) on success and failure.
MinIO is an S3-compatible object storage system that can be self-hosted.
"""

from datetime import datetime
from typing import Optional, cast

import litellm
from litellm._logging import print_verbose, verbose_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import StandardLoggingPayload


class MinioLogger(CustomLogger):
    def __init__(
        self,
        minio_endpoint_url=None,
        minio_bucket_name=None,
        minio_access_key_id=None,
        minio_secret_access_key=None,
        minio_region_name=None,
        minio_path=None,
        minio_use_ssl=True,
        minio_verify=None,
        minio_api_version=None,
        minio_session_token=None,
        minio_config=None,
        minio_use_team_prefix=False,
        **kwargs,
    ):
        """
        Initialize MinIO logger.
        
        Args:
            minio_endpoint_url: MinIO endpoint URL (e.g., 'http://localhost:9000')
            minio_bucket_name: Bucket name in MinIO
            minio_access_key_id: MinIO access key
            minio_secret_access_key: MinIO secret key
            minio_region_name: Region name (default: 'us-east-1')
            minio_path: Optional path prefix for objects
            minio_use_ssl: Whether to use SSL (default: True)
            minio_verify: SSL verification (default: None, can be False to disable)
            minio_api_version: S3 API version
            minio_session_token: Session token for temporary credentials
            minio_config: Boto3 config object
            minio_use_team_prefix: Whether to use team alias as prefix
        """
        import boto3

        try:
            verbose_logger.debug(
                f"in init minio logger - s3_callback_params {litellm.s3_callback_params}"
            )

            minio_use_team_prefix = False

            if litellm.s3_callback_params is not None:
                # read in .env variables - example os.environ/MINIO_BUCKET_NAME
                for key, value in litellm.s3_callback_params.items():
                    if isinstance(value, str) and value.startswith("os.environ/"):
                        litellm.s3_callback_params[key] = litellm.get_secret(value)
                # now set minio params from litellm.s3_callback_params
                minio_endpoint_url = litellm.s3_callback_params.get(
                    "s3_endpoint_url"
                )
                minio_bucket_name = litellm.s3_callback_params.get(
                    "s3_bucket_name"
                )
                minio_access_key_id = litellm.s3_callback_params.get(
                    "s3_aws_access_key_id"
                )
                minio_secret_access_key = litellm.s3_callback_params.get(
                    "s3_aws_secret_access_key"
                )
                minio_region_name = litellm.s3_callback_params.get(
                    "s3_region_name"
                )
                minio_path = litellm.s3_callback_params.get("s3_path")
                minio_use_ssl = litellm.s3_callback_params.get(
                    "s3_use_ssl", True
                )
                minio_verify = litellm.s3_callback_params.get("s3_verify")
                minio_api_version = litellm.s3_callback_params.get(
                    "s3_api_version"
                )
                minio_session_token = litellm.s3_callback_params.get(
                    "s3_aws_session_token"
                )
                minio_config = litellm.s3_callback_params.get("s3_config")
                # done reading litellm.s3_callback_params
                minio_use_team_prefix = bool(
                    litellm.s3_callback_params.get("s3_use_team_prefix", False)
                )
            
            self.minio_use_team_prefix = minio_use_team_prefix
            self.bucket_name = minio_bucket_name
            self.minio_path = minio_path
            
            verbose_logger.debug(f"minio logger using endpoint url {minio_endpoint_url}")
            
            # Create an S3 client configured for MinIO
            verbose_logger.debug(
                f"minio logger boto3.client args: endpoint_url={minio_endpoint_url}, "
                f"aws_access_key_id={'***' if minio_access_key_id else None}, "
                f"aws_secret_access_key={'***' if minio_secret_access_key else None}, "
                f"region_name={minio_region_name or 'us-east-1'}, "
                f"api_version={minio_api_version}, "
                f"use_ssl={minio_use_ssl}, "
                f"verify={minio_verify}, "
                f"aws_session_token={'***' if minio_session_token else None}, "
                f"config={minio_config}"
            )
            self.s3_client = boto3.client(
                "s3",
                endpoint_url=minio_endpoint_url,
                aws_access_key_id=minio_access_key_id,
                aws_secret_access_key=minio_secret_access_key,
                region_name=minio_region_name or "us-east-1",
                api_version=minio_api_version,
                use_ssl=minio_use_ssl,
                verify=minio_verify,
                aws_session_token=minio_session_token,
                config=minio_config,
                **kwargs,
            )
        except Exception as e:
            print_verbose(f"Got exception on init minio client {str(e)}")
            raise e

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        """
        Log successful events to MinIO.
        """
        self._log_event(
            kwargs=kwargs,
            response_obj=response_obj,
            start_time=start_time,
            end_time=end_time,
        )

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """
        Log failure events to MinIO.
        """
        self._log_event(
            kwargs=kwargs,
            response_obj=response_obj,
            start_time=start_time,
            end_time=end_time,
        )

    async def async_log_success_event(
        self, kwargs, response_obj, start_time, end_time
    ):
        """
        Async log successful events to MinIO.
        """
        self._log_event(
            kwargs=kwargs,
            response_obj=response_obj,
            start_time=start_time,
            end_time=end_time,
        )

    async def async_log_failure_event(
        self, kwargs, response_obj, start_time, end_time
    ):
        """
        Async log failure events to MinIO.
        """
        self._log_event(
            kwargs=kwargs,
            response_obj=response_obj,
            start_time=start_time,
            end_time=end_time,
        )

    def _log_event(self, kwargs, response_obj, start_time, end_time):
        """
        Core method to log events to MinIO.
        """
        try:
            verbose_logger.debug(
                f"MinIO Logging - Enters logging function for model {kwargs}"
            )

            # Get the standard logging payload
            payload: Optional[StandardLoggingPayload] = cast(
                Optional[StandardLoggingPayload],
                kwargs.get("standard_logging_object", None),
            )

            if payload is None:
                return

            team_alias = payload["metadata"].get("user_api_key_team_alias")

            team_alias_prefix = ""
            if (
                litellm.enable_preview_features
                and self.minio_use_team_prefix
                and team_alias is not None
            ):
                team_alias_prefix = f"{team_alias}/"

            s3_file_name = litellm.utils.get_logging_id(start_time, payload) or ""
            s3_object_key = self._get_minio_object_key(
                minio_path=cast(Optional[str], self.minio_path) or "",
                team_alias_prefix=team_alias_prefix,
                start_time=start_time,
                file_name=s3_file_name,
            )

            s3_object_download_filename = (
                "time-"
                + start_time.strftime("%Y-%m-%dT%H-%M-%S-%f")
                + "_"
                + payload["id"]
                + ".json"
            )

            from litellm.litellm_core_utils.safe_json_dumps import safe_dumps

            payload_str = safe_dumps(payload)

            print_verbose(f"\nMinIO Logger - Logging payload = {payload_str}")

            # Upload to MinIO using simple boto3 put_object
            response = self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_object_key,
                Body=payload_str,
                ContentType="application/json",
            )

            print_verbose(f"Response from MinIO: {str(response)}")
            print_verbose(f"MinIO Layer Logging - final response object: {response_obj}")
            
            return response
        except Exception as e:
            verbose_logger.exception(f"MinIO Layer Error - {str(e)}")
            pass

    def _get_minio_object_key(
        self,
        minio_path: str,
        team_alias_prefix: str,
        start_time: datetime,
        file_name: str,
    ) -> str:
        """
        Generate the object key for MinIO storage.
        
        Format: [path/][team_alias/]YYYY-MM-DD/filename.json
        """
        object_key = (
            (minio_path.rstrip("/") + "/" if minio_path else "")
            + team_alias_prefix
            + start_time.strftime("%Y-%m-%d")
            + "/"
            + file_name
        )
        object_key += ".json"
        return object_key

