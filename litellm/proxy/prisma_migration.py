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

if os.getenv("USE_AWS_KMS", None) is not None and os.getenv("USE_AWS_KMS") == "True":
    ## V2 IMPLEMENTATION OF AWS KMS - USER WANTS TO DECRYPT MULTIPLE KEYS IN THEIR ENV
    new_env_var = decrypt_env_var()

    for k, v in new_env_var.items():
        os.environ[k] = v

# Check if DATABASE_URL is not set
database_url = os.getenv("DATABASE_URL")
if not database_url:
    # Check if all required variables are provided
    database_host = os.getenv("DATABASE_HOST")
    database_username = os.getenv("DATABASE_USERNAME")
    database_password = os.getenv("DATABASE_PASSWORD")
    database_name = os.getenv("DATABASE_NAME")

    # Log the environment variables used for building the database URL
    print("Environment variables for database configuration:")
    print(f"DATABASE_HOST: {database_host}")
    print(f"DATABASE_USERNAME: {database_username}")
    print(f"DATABASE_PASSWORD: {database_password}")  # Mask password
    print(f"DATABASE_NAME: {database_name}")

    if database_host and database_username and database_password and database_name:
        # Construct DATABASE_URL from the provided variables
        database_url = f"postgresql://{database_username}:{database_password}@{database_host}/{database_name}"
        os.environ["DATABASE_URL"] = database_url
        print(f"Constructed DATABASE_URL: {database_url}")  # Log the constructed URL
    else:
        print(  # noqa
            "Error: Required database environment variables are not set. Provide a postgres url for DATABASE_URL."  # noqa
        )
        exit(1)
else:
    print(f"Using existing DATABASE_URL: {database_url}")  # Log existing DATABASE_URL

# Set DIRECT_URL to the value of DATABASE_URL if it is not set, required for migrations
direct_url = os.getenv("DIRECT_URL")
if not direct_url:
    os.environ["DIRECT_URL"] = database_url

# Apply migrations
retry_count = 0
max_retries = 100
exit_code = 1

disable_schema_update = os.getenv("DISABLE_SCHEMA_UPDATE")
if disable_schema_update is not None and disable_schema_update == "True":
    print("Skipping schema update...")  # noqa
    exit(0)

while retry_count < max_retries and exit_code != 0:
    retry_count += 1
    print(f"Attempt {retry_count}...")  # noqa

    # run prisma generate
    print("Running 'prisma generate'...")  # noqa
    result = subprocess.run(["prisma", "generate"], capture_output=True, text=True)
    print(f"'prisma generate' stdout: {result.stdout}")  # Log stdout
    print(f"'prisma generate' stderr: {result.stderr}")  # Log stderr
    exit_code = result.returncode

    if exit_code != 0:
        print(f"'prisma generate' failed with exit code {exit_code}.")  # noqa

    # Run the Prisma db push command
    print("Running 'prisma db push --accept-data-loss'...")  # noqa
    result = subprocess.run(
        ["prisma", "db", "push", "--accept-data-loss"],
        capture_output=True,
        text=True
    )
    exit_code = result.returncode

    if exit_code != 0:
        print(f"'prisma db push' stdout: {result.stdout}")  # Log stdout
        print(f"'prisma db push' stderr: {result.stderr}")  # Log stderr
        print(f"'prisma db push' failed with exit code {exit_code}.")  # noqa

    if exit_code != 0 and retry_count < max_retries:
        print("Retrying in 10 seconds...")  # noqa
        time.sleep(10)

if exit_code != 0:
    print(f"'prisma db push' stdout: {result.stdout}")  # Log stdout
    print(f"'prisma db push' stderr: {result.stderr}")  # Log stderr
    print(f"Unable to push database changes after {max_retries} retries.")  # noqa
    exit(1)

print("Database push successful!")  # noqa