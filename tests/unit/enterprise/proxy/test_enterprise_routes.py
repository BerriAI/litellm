"""
Test enterprise_routes imports work correctly

This validates that all imports can be resolved to prevent broken imports
from breaking the enterprise proxy initialization.
"""

import ast
import os

import pytest


def test_enterprise_routes_all_imports_exist():
    """
    Validate that all relative imports in enterprise_routes.py exist in the filesystem.
    
    This catches any import errors from moved/deleted modules without hardcoding
    specific module names. Works by checking that imported files actually exist.
    """
    # Path to the enterprise_routes.py source file
    enterprise_routes_path = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..", "..", 
        "enterprise", "litellm_enterprise", "proxy", "enterprise_routes.py"
    )
    
    enterprise_routes_path = os.path.normpath(enterprise_routes_path)
    enterprise_proxy_dir = os.path.dirname(enterprise_routes_path)
    
    if not os.path.exists(enterprise_routes_path):
        pytest.skip(f"Enterprise routes file not found at {enterprise_routes_path}")
    
    # Read and parse the source file
    with open(enterprise_routes_path, "r") as f:
        source_code = f.read()
    
    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        pytest.fail(f"Syntax error in enterprise_routes.py: {e}")
    
    # Check all relative imports
    missing_imports = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            # level > 0 means it's a relative import (. or .. etc)
            if node.level and node.level > 0:
                module = node.module or ""
                
                # Convert relative import to file path
                # e.g., "audit_logging_endpoints" -> "audit_logging_endpoints.py"
                # e.g., "vector_stores.endpoints" -> "vector_stores/endpoints.py"
                module_path = module.replace(".", os.sep) if module else ""
                
                # Check both .py file and package directory
                file_path = os.path.join(enterprise_proxy_dir, module_path + ".py") if module_path else None
                package_path = os.path.join(enterprise_proxy_dir, module_path, "__init__.py") if module_path else None
                
                # If module is empty (e.g., "from . import something"), skip check
                if not module:
                    continue
                
                file_exists = file_path and os.path.exists(file_path)
                package_exists = package_path and os.path.exists(package_path)
                
                if not file_exists and not package_exists:
                    missing_imports.append(
                        f"Line {node.lineno}: Cannot find '.{module}' "
                        f"(checked: {file_path} and {package_path})"
                    )
    
    if missing_imports:
        error_msg = "Found imports in enterprise_routes.py that don't exist:\n"
        error_msg += "\n".join(missing_imports)
        error_msg += "\n\nThis usually means a module was moved or deleted but the import wasn't updated."
        pytest.fail(error_msg)
