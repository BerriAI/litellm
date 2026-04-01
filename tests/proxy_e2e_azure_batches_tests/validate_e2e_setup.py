#!/usr/bin/env python
"""
Validation script for Azure Batch E2E test setup.
Run this before running the actual tests to verify all components are accessible.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath("../.."))

def check_imports():
    """Verify all required imports work."""
    print("Checking imports...")
    try:
        from base_integration_test import (
            get_mock_server_base_url,
            get_litellm_base_url,
            get_litellm_api_key,
        )
        print("  ✓ base_integration_test imports OK")
        
        from test_managed_files_base import ManagedFilesBase, get_batch_model_names
        print("  ✓ test_managed_files_base imports OK")
        
        from fixtures.mock_azure_batch_server import create_mock_azure_batch_server
        print("  ✓ mock_azure_batch_server imports OK")
        
        import httpx
        import openai
        import psycopg2
        import uvicorn
        print("  ✓ All external dependencies OK")
        
        return True
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return False


def check_config_file():
    """Verify config file exists."""
    print("\nChecking config file...")
    config_path = Path(__file__).parent / "fixtures" / "config.yml"
    if config_path.exists():
        print(f"  ✓ Config file found: {config_path}")
        return True
    else:
        print(f"  ✗ Config file not found: {config_path}")
        return False


def check_database():
    """Verify database connection."""
    print("\nChecking database connection...")
    try:
        import psycopg2
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="litellm",
            user="llmproxy",
            password="dbpassword9090",
        )
        conn.close()
        print("  ✓ Database connection OK")
        return True
    except Exception as e:
        print(f"  ✗ Database connection failed: {e}")
        print("    Start PostgreSQL with:")
        print("    docker run --name litellm-postgres -e POSTGRES_USER=llmproxy \\")
        print("      -e POSTGRES_PASSWORD=dbpassword9090 -e POSTGRES_DB=litellm \\")
        print("      -p 5432:5432 -d postgres:15")
        return False


def check_ports():
    """Check if required ports are available."""
    print("\nChecking ports...")
    import socket
    
    for port, name in [(4000, "LiteLLM Proxy"), (8090, "Mock Server")]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("localhost", port))
                print(f"  ✓ Port {port} ({name}) is available")
            except OSError:
                print(f"  ⚠ Port {port} ({name}) is in use (will reuse if healthy)")
    return True


def main():
    print("=" * 70)
    print("Azure Batch E2E Test Setup Validation")
    print("=" * 70)
    
    checks = [
        check_imports(),
        check_config_file(),
        check_database(),
        check_ports(),
    ]
    
    print("\n" + "=" * 70)
    if all(checks):
        print("✓ All checks passed! Ready to run E2E tests.")
        print("\nRun tests with:")
        print("  cd litellm")
        print("  export DATABASE_URL='postgresql://llmproxy:dbpassword9090@localhost:5432/litellm'")
        print("  poetry run pytest tests/proxy_e2e_azure_batches_tests/test_proxy_e2e_azure_batches.py -vv")
        return 0
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
