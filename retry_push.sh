#!/bin/bash

retry_count=0
max_retries=3
exit_code=1

until [ $retry_count -ge $max_retries ] || [ $exit_code -eq 0 ]
do
    retry_count=$((retry_count+1))
    echo "Attempt $retry_count..."

    # Run the Prisma db push command
    prisma db push

    exit_code=$?

    if [ $exit_code -ne 0 ] && [ $retry_count -lt $max_retries ]; then
        echo "Retrying in 10 seconds..."
        sleep 10
    fi
done

if [ $exit_code -ne 0 ]; then
    echo "Unable to push database changes after $max_retries retries."
    exit 1
fi

echo "Database push successful!"