"""
Test for /mcp route 307 redirect fix
See: https://github.com/BerriAI/litellm/issues/23688
"""
import os
import re


def test_mcp_route_fix_in_source_code():
    """
    Verify that the fix is present in the source code by checking
    that app.add_route("/mcp", ...) is called before app.mount("/mcp", ...)
    
    This fixes the issue where clients accessing /mcp (without trailing slash)
    would receive a 307 redirect to /mcp/, disrupting MCP client connections.
    
    See: https://github.com/BerriAI/litellm/issues/23688
    """
    # Get the path to the server.py file - use cwd as starting point
    server_file = "litellm/proxy/_experimental/mcp_server/server.py"
    
    assert os.path.exists(server_file), f"Server file not found: {server_file}"
    
    with open(server_file, "r") as f:
        content = f.read()
    
    # Check for the explicit route registration for /mcp
    add_route_pattern = r'app\.add_route\s*\(\s*["\']?/mcp["\']?'
    assert re.search(add_route_pattern, content), \
        "Fix not found: app.add_route('/mcp', ...) should be present in server.py"
    
    # Check for the issue reference comment
    assert "23688" in content, \
        "Issue reference #23688 should be present in the code comments"
    
    # Verify the route is added before the mount to ensure it takes precedence
    add_route_pos = content.find('app.add_route("/mcp"')
    mount_pos = content.find('app.mount("/mcp"')
    assert add_route_pos > 0 and mount_pos > 0 and add_route_pos < mount_pos, \
        "app.add_route('/mcp', ...) should come before app.mount('/mcp', ...)"


if __name__ == "__main__":
    test_mcp_route_fix_in_source_code()
    print("All tests passed!")
