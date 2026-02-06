## Tests that chat_completion endpoint has no imports inside function bodies
## This is critical for performance optimization in the hot path

import ast
from pathlib import Path


def test_chat_completion_no_imports():
    """Test that chat_completion endpoint has no imports in function bodies."""
    # Path to the proxy server file
    proxy_server_path = Path(__file__).parent.parent.parent / "litellm" / "proxy" / "proxy_server.py"
    
    with open(proxy_server_path, 'r') as f:
        content = f.read()
    
    # Parse the AST
    tree = ast.parse(content)
    
    # Find the chat_completion function
    chat_completion_func = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.AsyncFunctionDef) and node.name == "chat_completion"):
            chat_completion_func = node
            break
    
    assert chat_completion_func is not None, "chat_completion function not found"
    
    # Check for imports inside the function body
    import_violations = []
    
    for node in ast.walk(chat_completion_func):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            # Get line number
            line_num = node.lineno
            import_violations.append(line_num)
    
    # Assert no import violations found
    if import_violations:
        print(f"Found {len(import_violations)} import violations in chat_completion endpoint:")
        for line_num in import_violations:
            print(f"  - Line {line_num}: Import statement found")
        print("\nchat_completion endpoint should not contain imports for optimal performance.")
        raise Exception("Import violations found in chat_completion endpoint")