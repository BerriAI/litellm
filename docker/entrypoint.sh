#!/bin/bash
echo $(pwd)

# Run the Python migration script
python3 litellm/proxy/prisma_migration.py

# Check if the Python script executed successfully
if [ $? -eq 0 ]; then
    echo "Migration script ran successfully!"
else
    echo "Migration script failed!"
    exit 1
fi
