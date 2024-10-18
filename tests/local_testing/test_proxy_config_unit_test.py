import os
import sys
import traceback
from unittest import mock
import pytest

from dotenv import load_dotenv

import litellm.proxy
import litellm.proxy.proxy_server

load_dotenv()
import io
import os

# this file is to test litellm/proxy

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import asyncio
import logging

from litellm.proxy.proxy_server import ProxyConfig


@pytest.mark.asyncio
async def test_basic_reading_configs_from_files():
    """
    Test that the config is read correctly from the files in the example_config_yaml folder
    """
    proxy_config_instance = ProxyConfig()
    current_path = os.path.dirname(os.path.abspath(__file__))
    example_config_yaml_path = os.path.join(current_path, "example_config_yaml")

    # get all the files from example_config_yaml
    files = os.listdir(example_config_yaml_path)
    print(files)

    for file in files:
        config_path = os.path.join(example_config_yaml_path, file)
        config = await proxy_config_instance.get_config(config_file_path=config_path)
        print(config)


@pytest.mark.asyncio
async def test_read_config_from_bad_file_path():
    """
    Raise an exception if the file path is not valid
    """
    proxy_config_instance = ProxyConfig()
    config_path = "non-existent-file.yaml"
    with pytest.raises(Exception):
        config = await proxy_config_instance.get_config(config_file_path=config_path)


@pytest.mark.asyncio
async def test_read_config_file_with_os_environ_vars():
    """
    Ensures os.environ variables are read correctly from config.yaml
    Following vars are set as os.environ variables in the config.yaml file
    - DEFAULT_USER_ROLE
    - AWS_ACCESS_KEY_ID
    - AWS_SECRET_ACCESS_KEY
    - AZURE_GPT_4O
    - FIREWORKS
    """

    _env_vars_for_testing = {
        "DEFAULT_USER_ROLE": "admin",
        "AWS_ACCESS_KEY_ID": "1234567890",
        "AWS_SECRET_ACCESS_KEY": "1234567890",
        "AZURE_GPT_4O": "1234567890",
        "FIREWORKS": "1234567890",
    }

    _old_env_vars = {}
    for key, value in _env_vars_for_testing.items():
        if key in os.environ:
            _old_env_vars[key] = os.environ.get(key)
        os.environ[key] = value

    # Read config
    proxy_config_instance = ProxyConfig()
    current_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(
        current_path, "example_config_yaml", "config_with_env_vars.yaml"
    )
    config = await proxy_config_instance.get_config(config_file_path=config_path)
    print(config)

    # Add assertions
    assert (
        config["litellm_settings"]["default_internal_user_params"]["user_role"]
        == "admin"
    )
    assert (
        config["litellm_settings"]["s3_callback_params"]["s3_aws_access_key_id"]
        == "1234567890"
    )
    assert (
        config["litellm_settings"]["s3_callback_params"]["s3_aws_secret_access_key"]
        == "1234567890"
    )

    for model in config["model_list"]:
        if "azure" in model["litellm_params"]["model"]:
            assert model["litellm_params"]["api_key"] == "1234567890"
        elif "fireworks" in model["litellm_params"]["model"]:
            assert model["litellm_params"]["api_key"] == "1234567890"

    # cleanup
    for key, value in _env_vars_for_testing.items():
        if key in _old_env_vars:
            os.environ[key] = _old_env_vars[key]
        else:
            del os.environ[key]
