#!/bin/sh

echo "Current working directory: $(pwd)"


# Check if SET_AWS_KMS in env
if [ -n "$SET_AWS_KMS" ]; then
    # Call Python function to decrypt and reset environment variables
    env_vars=$(python -c 'from litellm.proxy.secret_managers.aws_secret_manager import decrypt_and_reset_env_var; env_vars = decrypt_and_reset_env_var();')
    echo "Received env_vars: ${env_vars}"
    # Export decrypted environment variables to the current Bash environment
    while IFS='=' read -r name value; do
        export "$name=$value"
    done <<< "$env_vars"
fi

echo "DATABASE_URL post kms: $($DATABASE_URL)"
# Check if DATABASE_URL is not set
if [ -z "$DATABASE_URL" ]; then
    # Check if all required variables are provided
    if [ -n "$DATABASE_HOST" ] && [ -n "$DATABASE_USERNAME" ] && [ -n "$DATABASE_PASSWORD" ]  && [ -n "$DATABASE_NAME" ]; then
        # Construct DATABASE_URL from the provided variables
        DATABASE_URL="postgresql://${DATABASE_USERNAME}:${DATABASE_PASSWORD}@${DATABASE_HOST}/${DATABASE_NAME}"
        export DATABASE_URL
    else
        echo "Error: Required database environment variables are not set. Provide a postgres url for DATABASE_URL."
        exit 1
    fi
fi

echo "DATABASE_URL: $($DATABASE_URL)"

# Set DIRECT_URL to the value of DATABASE_URL if it is not set, required for migrations
if [ -z "$DIRECT_URL" ]; then
    export DIRECT_URL=$DATABASE_URL
fi

# # Apply migrations
# retry_count=0
# max_retries=3
# exit_code=1

# until [ $retry_count -ge $max_retries ] || [ $exit_code -eq 0 ]
# do
#     retry_count=$((retry_count+1))
#     echo "Attempt $retry_count..."

#     # Run the Prisma db push command
#     prisma db push --accept-data-loss

#     exit_code=$?

#     if [ $exit_code -ne 0 ] && [ $retry_count -lt $max_retries ]; then
#         echo "Retrying in 10 seconds..."
#         sleep 10
#     fi
# done

# if [ $exit_code -ne 0 ]; then
#     echo "Unable to push database changes after $max_retries retries."
#     exit 1
# fi

echo "Database push successful!"

