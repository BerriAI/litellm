from prisma import Prisma
import csv
import json
import asyncio
from datetime import datetime
from typing import Optional, List, Dict, Any

import os

## VARIABLES
DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/litellm"
CSV_FILE_PATH = "./path_to_csv.csv"

os.environ["DATABASE_URL"] = DATABASE_URL


async def parse_csv_value(value: str, field_type: str) -> Any:
    """Parse CSV values according to their expected types"""
    if value == "NULL" or value == "" or value is None:
        return None

    if field_type == "boolean":
        return value.lower() == "true"
    elif field_type == "float":
        return float(value)
    elif field_type == "int":
        return int(value) if value.isdigit() else None
    elif field_type == "bigint":
        return int(value) if value.isdigit() else None
    elif field_type == "datetime":
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except:
            return None
    elif field_type == "json":
        try:
            return value if value else json.dumps({})
        except:
            return json.dumps({})
    elif field_type == "string_array":
        # Handle string arrays like {default-models}
        if value.startswith("{") and value.endswith("}"):
            content = value[1:-1]  # Remove braces
            if content:
                return [item.strip() for item in content.split(",")]
            else:
                return []
        return []
    else:
        return value


async def migrate_verification_tokens():
    """Main migration function"""
    prisma = Prisma()
    await prisma.connect()

    try:
        # Read CSV file
        csv_file_path = CSV_FILE_PATH

        with open(csv_file_path, "r", encoding="utf-8") as file:
            csv_reader = csv.DictReader(file)

            processed_count = 0
            error_count = 0

            for row in csv_reader:
                try:
                    # Replace 'default-team' with the specified UUID
                    team_id = row.get("team_id")
                    if team_id == "NULL" or team_id == "":
                        team_id = None

                    # Prepare data for insertion
                    verification_token_data = {
                        "token": row["token"],
                        "key_name": await parse_csv_value(row["key_name"], "string"),
                        "key_alias": await parse_csv_value(row["key_alias"], "string"),
                        "soft_budget_cooldown": await parse_csv_value(
                            row["soft_budget_cooldown"], "boolean"
                        ),
                        "spend": await parse_csv_value(row["spend"], "float"),
                        "expires": await parse_csv_value(row["expires"], "datetime"),
                        "models": await parse_csv_value(row["models"], "string_array"),
                        "aliases": await parse_csv_value(row["aliases"], "json"),
                        "config": await parse_csv_value(row["config"], "json"),
                        "user_id": await parse_csv_value(row["user_id"], "string"),
                        "team_id": team_id,
                        "permissions": await parse_csv_value(
                            row["permissions"], "json"
                        ),
                        "max_parallel_requests": await parse_csv_value(
                            row["max_parallel_requests"], "int"
                        ),
                        "metadata": await parse_csv_value(row["metadata"], "json"),
                        "tpm_limit": await parse_csv_value(row["tpm_limit"], "bigint"),
                        "rpm_limit": await parse_csv_value(row["rpm_limit"], "bigint"),
                        "max_budget": await parse_csv_value(row["max_budget"], "float"),
                        "budget_duration": await parse_csv_value(
                            row["budget_duration"], "string"
                        ),
                        "budget_reset_at": await parse_csv_value(
                            row["budget_reset_at"], "datetime"
                        ),
                        "allowed_cache_controls": await parse_csv_value(
                            row["allowed_cache_controls"], "string_array"
                        ),
                        "model_spend": await parse_csv_value(
                            row["model_spend"], "json"
                        ),
                        "model_max_budget": await parse_csv_value(
                            row["model_max_budget"], "json"
                        ),
                        "budget_id": await parse_csv_value(row["budget_id"], "string"),
                        "blocked": await parse_csv_value(row["blocked"], "boolean"),
                        "created_at": await parse_csv_value(
                            row["created_at"], "datetime"
                        ),
                        "updated_at": await parse_csv_value(
                            row["updated_at"], "datetime"
                        ),
                        "allowed_routes": await parse_csv_value(
                            row["allowed_routes"], "string_array"
                        ),
                        "object_permission_id": await parse_csv_value(
                            row["object_permission_id"], "string"
                        ),
                        "created_by": await parse_csv_value(
                            row["created_by"], "string"
                        ),
                        "updated_by": await parse_csv_value(
                            row["updated_by"], "string"
                        ),
                        "organization_id": await parse_csv_value(
                            row["organization_id"], "string"
                        ),
                    }

                    # Remove None values to use database defaults
                    verification_token_data = {
                        k: v
                        for k, v in verification_token_data.items()
                        if v is not None
                    }

                    # Check if token already exists
                    existing_token = await prisma.litellm_verificationtoken.find_unique(
                        where={"token": verification_token_data["token"]}
                    )

                    if existing_token:
                        print(
                            f"Token {verification_token_data['token']} already exists, skipping..."
                        )
                        continue

                    # Insert the record
                    await prisma.litellm_verificationtoken.create(
                        data=verification_token_data
                    )

                    processed_count += 1
                    print(
                        f"Successfully migrated token: {verification_token_data['token']}"
                    )

                except Exception as e:
                    error_count += 1
                    print(
                        f"Error processing row with token {row.get('token', 'unknown')}: {str(e)}"
                    )
                    continue

            print(f"\nMigration completed!")
            print(f"Successfully processed: {processed_count} records")
            print(f"Errors encountered: {error_count} records")

    except Exception as e:
        print(f"Migration failed: {str(e)}")

    finally:
        await prisma.disconnect()


if __name__ == "__main__":
    asyncio.run(migrate_verification_tokens())
