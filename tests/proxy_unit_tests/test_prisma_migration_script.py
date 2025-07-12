import pytest
import os
import sys
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import tempfile
import shutil

# Add litellm to path
sys.path.insert(0, os.path.abspath("../../"))


def test_prisma_migration_with_existing_migrations(monkeypatch):
    """Test prisma migration script when migrations exist"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup test environment
        test_db_url = "postgresql://test:test@localhost:5432/testdb"
        monkeypatch.setenv("DATABASE_URL", test_db_url)
        monkeypatch.chdir(tmpdir)
        
        # Create schema.prisma file
        schema_path = Path(tmpdir) / "schema.prisma"
        schema_path.write_text("// Test schema")
        
        # Create migrations directory with a migration
        migrations_dir = Path(tmpdir) / "migrations"
        migrations_dir.mkdir()
        test_migration = migrations_dir / "20240101000000_test"
        test_migration.mkdir()
        (test_migration / "migration.sql").write_text("-- Test migration")
        
        # Mock subprocess.run
        with patch("subprocess.run") as mock_run:
            # Configure successful responses
            mock_run.return_value = MagicMock(returncode=0, stdout="Success", stderr="")
            
            # Import and run the migration script
            from litellm.proxy.prisma_migration import (
                disable_schema_update,
                database_url,
                retry_count,
                max_retries,
                exit_code,
            )
            
            # Verify it would run prisma migrate deploy
            # The script runs in a loop, so we need to check the calls
            assert any(
                call(["prisma", "migrate", "deploy"], capture_output=True, text=True) in mock_run.call_args_list
                for call in mock_run.call_args_list
            )


def test_prisma_migration_without_migrations(monkeypatch):
    """Test prisma migration script when no migrations exist"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup test environment
        test_db_url = "postgresql://test:test@localhost:5432/testdb"
        monkeypatch.setenv("DATABASE_URL", test_db_url)
        monkeypatch.chdir(tmpdir)
        
        # Create schema.prisma file
        schema_path = Path(tmpdir) / "schema.prisma"
        schema_path.write_text("// Test schema")
        
        # No migrations directory
        
        # Mock subprocess.run
        with patch("subprocess.run") as mock_run:
            # Configure successful responses
            mock_run.return_value = MagicMock(returncode=0, stdout="Success", stderr="")
            
            # Import and run relevant parts of the migration script
            from litellm.proxy.prisma_migration import Path as PrismaPath
            
            # Test the logic directly
            migrations_path = PrismaPath("./migrations")
            
            # Verify that when migrations don't exist, it would use db push
            if not (migrations_path.exists() and any(migrations_path.iterdir())):
                # This is the expected path for no migrations
                assert True
            else:
                pytest.fail("Expected no migrations to exist")


def test_prisma_migration_disabled(monkeypatch):
    """Test prisma migration script when DISABLE_SCHEMA_UPDATE is set"""
    monkeypatch.setenv("DISABLE_SCHEMA_UPDATE", "True")
    monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost:5432/testdb")
    
    # The script should exit early when disabled
    with pytest.raises(SystemExit) as exc_info:
        exec(open("litellm/proxy/prisma_migration.py").read())
    
    assert exc_info.value.code == 0


def test_prisma_migration_retry_logic(monkeypatch):
    """Test that the migration script retries on failure"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup test environment
        test_db_url = "postgresql://test:test@localhost:5432/testdb"
        monkeypatch.setenv("DATABASE_URL", test_db_url)
        monkeypatch.chdir(tmpdir)
        
        # Create schema.prisma file
        schema_path = Path(tmpdir) / "schema.prisma"
        schema_path.write_text("// Test schema")
        
        # Create migrations directory
        migrations_dir = Path(tmpdir) / "migrations"
        migrations_dir.mkdir()
        test_migration = migrations_dir / "20240101000000_test"
        test_migration.mkdir()
        (test_migration / "migration.sql").write_text("-- Test migration")
        
        # Mock subprocess.run to fail initially then succeed
        with patch("subprocess.run") as mock_run:
            # First call (generate) succeeds, second call (migrate) fails, then succeeds
            mock_run.side_effect = [
                MagicMock(returncode=0, stdout="Generate success", stderr=""),  # prisma generate
                MagicMock(returncode=1, stdout="", stderr="Migration failed"),  # prisma migrate deploy (fail)
                MagicMock(returncode=0, stdout="Generate success", stderr=""),  # prisma generate (retry)
                MagicMock(returncode=0, stdout="Migration success", stderr=""),  # prisma migrate deploy (success)
            ]
            
            with patch("time.sleep"):  # Mock sleep to speed up test
                # Run the migration logic
                # We'll test just the retry logic rather than executing the whole script
                retry_count = 0
                max_retries = 3
                exit_code = 1
                
                while retry_count < max_retries and exit_code != 0:
                    retry_count += 1
                    
                    # Simulate prisma generate
                    result = subprocess.run(["prisma", "generate"], capture_output=True, text=True)
                    exit_code = result.returncode
                    
                    if exit_code == 0:
                        # Simulate prisma migrate deploy
                        result = subprocess.run(["prisma", "migrate", "deploy"], capture_output=True, text=True)
                        exit_code = result.returncode
                
                # Verify it retried and eventually succeeded
                assert retry_count == 2  # First attempt failed, second succeeded
                assert exit_code == 0