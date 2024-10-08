# What is this?
## Unit tests for 'docker/entrypoint.sh'

import pytest
import sys
import os

sys.path.insert(
    0, os.path.abspath("../")
)  # Adds the parent directory to the system path
import litellm
import subprocess


@pytest.mark.skip(reason="local test")
def test_decrypt_and_reset_env():
    os.environ["DATABASE_URL"] = (
        "aws_kms/AQICAHgwddjZ9xjVaZ9CNCG8smFU6FiQvfdrjL12DIqi9vUAQwHwF6U7caMgHQa6tK+TzaoMAAAAzjCBywYJKoZIhvcNAQcGoIG9MIG6AgEAMIG0BgkqhkiG9w0BBwEwHgYJYIZIAWUDBAEuMBEEDCmu+DVeKTm5tFZu6AIBEICBhnOFQYviL8JsciGk0bZsn9pfzeYWtNkVXEsl01AdgHBqT9UOZOI4ZC+T3wO/fXA7wdNF4o8ASPDbVZ34ZFdBs8xt4LKp9niufL30WYBkuuzz89ztly0jvE9pZ8L6BMw0ATTaMgIweVtVSDCeCzEb5PUPyxt4QayrlYHBGrNH5Aq/axFTe0La"
    )
    from litellm.secret_managers.aws_secret_manager import (
        decrypt_and_reset_env_var,
    )

    decrypt_and_reset_env_var()

    assert os.environ["DATABASE_URL"] is not None
    assert isinstance(os.environ["DATABASE_URL"], str)
    assert not os.environ["DATABASE_URL"].startswith("aws_kms/")

    print("DATABASE_URL={}".format(os.environ["DATABASE_URL"]))


@pytest.mark.skip(reason="local test")
def test_entrypoint_decrypt_and_reset():
    os.environ["DATABASE_URL"] = (
        "aws_kms/AQICAHgwddjZ9xjVaZ9CNCG8smFU6FiQvfdrjL12DIqi9vUAQwHwF6U7caMgHQa6tK+TzaoMAAAAzjCBywYJKoZIhvcNAQcGoIG9MIG6AgEAMIG0BgkqhkiG9w0BBwEwHgYJYIZIAWUDBAEuMBEEDCmu+DVeKTm5tFZu6AIBEICBhnOFQYviL8JsciGk0bZsn9pfzeYWtNkVXEsl01AdgHBqT9UOZOI4ZC+T3wO/fXA7wdNF4o8ASPDbVZ34ZFdBs8xt4LKp9niufL30WYBkuuzz89ztly0jvE9pZ8L6BMw0ATTaMgIweVtVSDCeCzEb5PUPyxt4QayrlYHBGrNH5Aq/axFTe0La"
    )
    command = "./docker/entrypoint.sh"
    directory = ".."  # Relative to the current directory

    # Run the command using subprocess
    result = subprocess.run(
        command, shell=True, cwd=directory, capture_output=True, text=True
    )

    # Print the output for debugging purposes
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    # Assert the script ran successfully
    assert result.returncode == 0, "The shell script did not execute successfully"
    assert (
        "DECRYPTS VALUE" in result.stdout
    ), "Expected output not found in script output"
    assert (
        "Database push successful!" in result.stdout
    ), "Expected output not found in script output"

    assert False
