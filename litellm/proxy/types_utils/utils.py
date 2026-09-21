import asyncio
import importlib
import importlib.util
import os
from collections.abc import Callable
from typing import Any, Final, Literal, get_type_hints


def get_instance_fn(value: str, config_file_path: str | None = None) -> Any:
    module_name = value
    instance_name = None
    try:
        # Check if value starts with s3:// or gcs://
        if value.startswith("s3://") or value.startswith("gcs://"):
            # Remote module loading is a documented operator feature when
            # invoked from config-file load (``config_file_path`` carries
            # the YAML path). Without that signal the URL is request-body
            # data on an admin endpoint — a one-step admin-to-RCE primitive
            # via ``_load_instance_from_remote_storage``'s ``exec_module``.
            # Register the module under ``litellm_settings`` in the
            # config.yaml instead.
            if config_file_path is None:
                raise ValueError(
                    "Remote module loading (s3://, gcs://) is only "
                    "permitted from the config-file load path. Register "
                    "the module under ``litellm_settings`` in your "
                    "config.yaml instead."
                )
            return _load_instance_from_remote_storage(value, config_file_path)

        # Split the path by dots to separate module from instance
        parts: Final = value.split(".")

        # The module path is all but the last part, and the instance_name is the last part
        module_name = ".".join(parts[:-1])
        instance_name = parts[-1]

        module_file_path = None
        if config_file_path is not None:
            directory: Final = os.path.dirname(config_file_path)
            module_file_path = os.path.join(directory, *module_name.split(".")) + ".py"

        if module_file_path is not None and os.path.exists(module_file_path):
            spec: Final = importlib.util.spec_from_file_location(module_name, module_file_path)
            if spec is None:
                raise ImportError(f"Could not find a module specification for {module_file_path}")
            module = importlib.util.module_from_spec(spec)
            if spec.loader is None:
                raise ImportError(f"Could not find a module loader for {module_file_path}")
            spec.loader.exec_module(module)
        else:
            module = importlib.import_module(module_name)

        # Get the instance from the module
        instance: Final = getattr(module, instance_name)

        return instance
    except ImportError as e:
        # Re-raise the exception with a user-friendly message
        if instance_name and module_name:
            raise ImportError(f"Could not import {instance_name} from {module_name}") from e
        else:
            raise e
    except Exception as e:
        raise e


def _load_instance_from_remote_storage(remote_url: str, config_file_path: str | None = None) -> Any:
    """
    Load custom logger instance from S3 or GCS URL.

    Expected format:
    - s3://bucket-name/path/to/module.instance_name
    - gcs://bucket-name/path/to/module.instance_name

    Args:
        remote_url (str): The s3:// or gcs:// URL
        config_file_path (str): Optional config file path for temp directory context

    Returns:
        Any: The loaded instance
    """
    try:
        from litellm._logging import verbose_proxy_logger

        # Parse the URL
        if remote_url.startswith("s3://"):
            storage_type = "s3"
            url_without_prefix = remote_url[5:]  # Remove 's3://'
        elif remote_url.startswith("gcs://"):
            storage_type = "gcs"
            url_without_prefix = remote_url[6:]  # Remove 'gcs://'
        else:
            raise ValueError(f"Unsupported URL scheme in {remote_url}")

        # Split bucket and path
        parts: Final = url_without_prefix.split("/", 1)
        if len(parts) < 2:
            raise ValueError(
                f"Invalid URL format: {remote_url}. Expected: {storage_type}://bucket-name/path/to/module.instance"
            )

        bucket_name: Final = parts[0]
        path_and_module: Final = parts[1]

        # Extract module path and instance name
        # Example: "loggers/custom_callbacks.proxy_handler_instance"
        # Handle case where user accidentally includes .py extension
        if path_and_module.endswith(".py"):
            module_name_without_py: Final = path_and_module[:-3]  # Remove .py
            raise ValueError(
                f"Invalid URL format in {remote_url}. "
                f"Don't include '.py' extension and you must specify the instance name. "
                f"Expected format: {storage_type}://{bucket_name}/{module_name_without_py}.instance_name "
                f"(e.g., {storage_type}://{bucket_name}/{module_name_without_py}.proxy_handler_instance)"
            )

        # Split by last dot to separate module from instance
        module_parts: Final = path_and_module.split(".")
        if len(module_parts) < 2:
            raise ValueError(f"Invalid module specification in {remote_url}. Expected: path/to/module.instance_name")

        instance_name: Final = module_parts[-1]
        module_path: Final = ".".join(module_parts[:-1])

        # Create object key (file path in bucket)
        object_key: Final = f"{module_path}.py"

        verbose_proxy_logger.debug(
            "Loading custom logger from %s: bucket=%s, object_key=%s, instance=%s",
            storage_type,
            bucket_name,
            object_key,
            instance_name,
        )

        import tempfile

        # Create temporary file for the downloaded module using the actual module name
        temp_file: Final = tempfile.NamedTemporaryFile(suffix=".py", delete=False)
        local_file_path: Final = temp_file.name
        temp_file.close()  # Close the file so we can write to it

        # Download the file
        if storage_type == "s3":
            from litellm.proxy.common_utils.load_config_utils import (
                download_python_file_from_s3,
            )

            success = download_python_file_from_s3(
                bucket_name=bucket_name,
                object_key=object_key,
                local_file_path=local_file_path,
            )
        else:  # gcs
            success = asyncio.run(_download_gcs_file_wrapper(bucket_name, object_key, local_file_path))

        if not success:
            raise ImportError(f"Failed to download {object_key} from {storage_type} bucket {bucket_name}")

        # Load the module from the downloaded file using the actual module name
        spec: Final = importlib.util.spec_from_file_location(module_path, local_file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create module spec for {local_file_path}")

        module: Final = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Get the instance
        instance: Final = getattr(module, instance_name)

        # Clean up the temporary file
        try:
            os.remove(local_file_path)
        except Exception as cleanup_error:
            verbose_proxy_logger.warning("Could not clean up temporary file %s: %s", local_file_path, cleanup_error)

        verbose_proxy_logger.info("Successfully loaded custom logger from %s", remote_url)
        return instance

    except Exception as e:
        raise ImportError(f"Failed to load custom logger from {remote_url}: {e}") from e


async def _download_gcs_file_wrapper(bucket_name: str, object_key: str, local_file_path: str) -> bool:
    """Wrapper for GCS download to handle async properly"""
    try:
        from litellm.proxy.common_utils.load_config_utils import (
            download_python_file_from_gcs,
        )

        return await download_python_file_from_gcs(bucket_name, object_key, local_file_path)
    except Exception as e:
        from litellm._logging import verbose_proxy_logger

        verbose_proxy_logger.error("Error downloading from GCS: %s", e)
        return False


def validate_custom_validate_return_type(
    fn: Callable[..., Any] | None,
) -> Callable[..., Literal[True]] | None:
    if fn is None:
        return None

    hints: Final = get_type_hints(fn)
    return_type: Final = hints.get("return")

    if return_type != Literal[True]:
        raise TypeError(f"Custom validator must be annotated to return Literal[True], got {return_type}")

    return fn
