import subprocess
from pathlib import Path
from datetime import datetime


def create_baseline():
    """Create baseline migration in deploy/migrations"""
    try:
        # Get paths
        root_dir = Path(__file__).parent.parent
        deploy_dir = root_dir / "deploy"
        migrations_dir = deploy_dir / "migrations"
        schema_path = root_dir / "schema.prisma"

        # Create migrations directory
        migrations_dir.mkdir(parents=True, exist_ok=True)

        # Create migration_lock.toml if it doesn't exist
        lock_file = migrations_dir / "migration_lock.toml"
        if not lock_file.exists():
            lock_file.write_text('provider = "postgresql"\n')

        # Create timestamp-based migration directory
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        migration_dir = migrations_dir / f"{timestamp}_baseline"
        migration_dir.mkdir(parents=True, exist_ok=True)

        # Generate migration SQL
        result = subprocess.run(
            [
                "prisma",
                "migrate",
                "diff",
                "--from-empty",
                "--to-schema-datamodel",
                str(schema_path),
                "--script",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        # Write the SQL to migration.sql
        migration_file = migration_dir / "migration.sql"
        migration_file.write_text(result.stdout)

        print(f"Created baseline migration in {migration_dir}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"Error running prisma command: {e.stderr}")
        return False
    except Exception as e:
        print(f"Error creating baseline migration: {str(e)}")
        return False


if __name__ == "__main__":
    create_baseline()
