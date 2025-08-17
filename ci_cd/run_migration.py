import os
import subprocess
from pathlib import Path
from datetime import datetime
import testing.postgresql
import shutil


def create_migration(migration_name: str = None):
    """
    Create a new migration SQL file in the migrations directory by comparing
    current database state with schema

    Args:
        migration_name (str): Name for the migration
    """
    try:
        # Get paths
        root_dir = Path(__file__).parent.parent
        migrations_dir = root_dir / "litellm-proxy-extras" / "litellm_proxy_extras" / "migrations"
        schema_path = root_dir / "schema.prisma"

        # Create temporary PostgreSQL database
        with testing.postgresql.Postgresql() as postgresql:
            db_url = postgresql.url()

            # Create temporary migrations directory next to schema.prisma
            temp_migrations_dir = schema_path.parent / "migrations"

            try:
                # Copy existing migrations to temp directory
                if temp_migrations_dir.exists():
                    shutil.rmtree(temp_migrations_dir)
                shutil.copytree(migrations_dir, temp_migrations_dir)

                # Apply existing migrations to temp database
                os.environ["DATABASE_URL"] = db_url
                subprocess.run(
                    ["prisma", "migrate", "deploy", "--schema", str(schema_path)],
                    check=True,
                )

                # Generate diff between current database and schema
                result = subprocess.run(
                    [
                        "prisma",
                        "migrate",
                        "diff",
                        "--from-url",
                        db_url,
                        "--to-schema-datamodel",
                        str(schema_path),
                        "--script",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )

                if result.stdout.strip():
                    # Generate timestamp and create migration directory
                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    migration_name = migration_name or "unnamed_migration"
                    migration_dir = migrations_dir / f"{timestamp}_{migration_name}"
                    migration_dir.mkdir(parents=True, exist_ok=True)

                    # Write the SQL to migration.sql
                    migration_file = migration_dir / "migration.sql"
                    migration_file.write_text(result.stdout)

                    print(f"Created migration in {migration_dir}")
                    return True
                else:
                    print("No schema changes detected. Migration not needed.")
                    return False

            finally:
                # Clean up: remove temporary migrations directory
                if temp_migrations_dir.exists():
                    shutil.rmtree(temp_migrations_dir)

    except subprocess.CalledProcessError as e:
        print(f"Error generating migration: {e.stderr}")
        return False
    except Exception as e:
        print(f"Error creating migration: {str(e)}")
        return False


if __name__ == "__main__":
    # If running directly, can optionally pass migration name as argument
    import sys

    migration_name = sys.argv[1] if len(sys.argv) > 1 else None
    create_migration(migration_name)
