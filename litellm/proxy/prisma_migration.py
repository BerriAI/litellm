# What is this?
## Script to apply initial prisma migration on Docker setup

import os
import subprocess
import sys
import time

sys.path.insert(
    0, os.path.abspath("./")
)  # Adds the parent directory to the system path
from litellm.secret_managers.aws_secret_manager import decrypt_env_var
from litellm._logging import verbose_proxy_logger

if os.getenv("USE_AWS_KMS", None) is not None and os.getenv("USE_AWS_KMS") == "True":
    ## V2 IMPLEMENTATION OF AWS KMS - USER WANTS TO DECRYPT MULTIPLE KEYS IN THEIR ENV
    new_env_var = decrypt_env_var()

    for k, v in new_env_var.items():
        os.environ[k] = v

# Check if DATABASE_URL is not set
database_url = os.getenv("DATABASE_URL")
if not database_url:
    verbose_proxy_logger.info("Constructing DATABASE_URL from environment variables")
    # Check if all required variables are provided
    database_host = os.getenv("DATABASE_HOST")
    database_username = os.getenv("DATABASE_USERNAME")
    database_password = os.getenv("DATABASE_PASSWORD")
    database_name = os.getenv("DATABASE_NAME")

    if database_host and database_username and database_password and database_name:
        # Construct DATABASE_URL from the provided variables
        database_url = f"postgresql://{database_username}:{database_password}@{database_host}/{database_name}"
        os.environ["DATABASE_URL"] = database_url  # Log the constructed URL
    else:
        verbose_proxy_logger.error(
            "Error: Required database environment variables are not set. Provide a postgres url for DATABASE_URL."  # noqa
        )
        exit(1)
else:
    verbose_proxy_logger.info("Using existing DATABASE_URL environment variable")  # Log existing DATABASE_URL

# Set DIRECT_URL to the value of DATABASE_URL if it is not set, required for migrations
direct_url = os.getenv("DIRECT_URL")
if not direct_url:
    os.environ["DIRECT_URL"] = database_url

# Apply migrations
retry_count = 0
max_retries = 3
exit_code = 1

disable_schema_update = os.getenv("DISABLE_SCHEMA_UPDATE")
if disable_schema_update is not None and disable_schema_update == "True":
    verbose_proxy_logger.info("Skipping schema update...")
    exit(0)

while retry_count < max_retries and exit_code != 0:
    retry_count += 1
    verbose_proxy_logger.info(f"Attempt {retry_count}...")

    # run prisma generate
    verbose_proxy_logger.info("Running 'prisma generate'...")
    result = subprocess.run(["prisma", "generate"], capture_output=True, text=True)
    verbose_proxy_logger.info(f"'prisma generate' stdout: {result.stdout}")  # Log stdout
    exit_code = result.returncode

    if exit_code != 0:
        verbose_proxy_logger.info(f"'prisma generate' failed with exit code {exit_code}.")
        verbose_proxy_logger.error(f"'prisma generate' stderr: {result.stderr}")  # Log stderr

    # Run the Prisma db push command
    verbose_proxy_logger.info("Running 'prisma db push --accept-data-loss'...")
    result = subprocess.run(
        ["prisma", "db", "push", "--accept-data-loss"], capture_output=True, text=True
    )
    verbose_proxy_logger.info(f"'prisma db push' stdout: {result.stdout}")  # Log stdout
    exit_code = result.returncode

    if exit_code != 0:
        verbose_proxy_logger.info(f"'prisma db push' stderr: {result.stderr}")  # Log stderr
        verbose_proxy_logger.error(f"'prisma db push' failed with exit code {exit_code}.")
        if retry_count < max_retries:
            verbose_proxy_logger.info("Retrying in 10 seconds...")
            time.sleep(10)

if retry_count == max_retries and exit_code != 0:
    verbose_proxy_logger.error(f"Unable to push database changes after {max_retries} retries.")
    exit(1)

verbose_proxy_logger.info("Database push successful!")
