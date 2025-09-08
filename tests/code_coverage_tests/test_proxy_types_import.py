import ast
import os
import sys


def test_proxy_types_not_imported():
    """
    Test that proxy._types is not directly imported in litellm/__init__.py
    by examining the source code using AST parsing.
    """
    # Read the litellm/__init__.py file
    # local_init_file = "../litellm/"
    init_file_path = os.path.join("./litellm", "__init__.py")
    if not os.path.exists(init_file_path):
        raise Exception(f"Could not find {init_file_path}")
    
    with open(init_file_path, "r") as f:
        content = f.read()
        lines = content.splitlines()  # Get lines for line number reporting
    
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        raise Exception(f"Could not parse {init_file_path}: {e}")
    
    # Check for direct imports of proxy._types
    found_imports = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if "proxy._types" in alias.name or "proxy/_types" in alias.name:
                    line_num = node.lineno
                    line_content = lines[line_num - 1] if line_num <= len(lines) else "Unknown"
                    import_statement = f"import {alias.name}"
                    found_imports.append({
                        'type': 'import',
                        'line': line_num,
                        'content': line_content.strip(),
                        'statement': import_statement,
                        'module': alias.name
                    })
                    
        elif isinstance(node, ast.ImportFrom):
            if node.module and ("proxy._types" in node.module or "proxy/_types" in node.module):
                line_num = node.lineno
                line_content = lines[line_num - 1] if line_num <= len(lines) else "Unknown"
                import_names = [alias.name for alias in node.names]
                import_statement = f"from {node.module} import {', '.join(import_names)}"
                found_imports.append({
                    'type': 'from_import',
                    'line': line_num,
                    'content': line_content.strip(),
                    'statement': import_statement,
                    'module': node.module
                })
    
    if found_imports:
        print("❌ BAD, this can import time to import litellm. Found direct imports of proxy._types in litellm/__init__.py:")
        print("=" * 80)
        for imp in found_imports:
            print(f"Line {imp['line']}: {imp['content']}")
            print(f"  Type: {imp['type']}")
            print(f"  Statement: {imp['statement']}")
            print(f"  Module: {imp['module']}")
            print("-" * 80)
        print("To fix this, please conditionally import this TYPE using TYPE_CHECKING")
        
        raise Exception(
            f"Found {len(found_imports)} direct import(s) of proxy._types in litellm/__init__.py"
        )
    
    print("✓ No direct imports of proxy._types found in litellm/__init__.py")
    return True


def main():
    """
    Main function to run the import test
    """
    print("=" * 60)
    print("Testing litellm import performance")
    print("Checking that proxy._types is not directly imported from litellm/__init__.py")
    print("=" * 60)
    
    try:
        test_proxy_types_not_imported()
        print("\n" + "=" * 60)
        print("✓ Test passed! proxy._types is not directly imported from litellm/__init__.py")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main() 