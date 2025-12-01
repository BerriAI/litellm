"""
Load testing script for MCP (Model Context Protocol) endpoints in LiteLLM Proxy.

This script is similar to no_cache_hits.py but tests MCP functionality instead.
It can be used with Locust to measure latency and performance of MCP operations.
"""
import os
import uuid
from locust import HttpUser, task, between, events

# Custom metric to track LiteLLM overhead duration for MCP operations
mcp_overhead_durations = []

@events.request.add_listener
def on_request(**kwargs):
    """Track LiteLLM overhead duration from response headers"""
    response = kwargs.get('response')
    if response and hasattr(response, 'headers') and response.headers:
        overhead_duration = response.headers.get('x-litellm-overhead-duration-ms')
        if overhead_duration:
            try:
                duration_ms = float(overhead_duration)
                mcp_overhead_durations.append(duration_ms)
                # Report as custom metric
                events.request.fire(
                    request_type="Custom",
                    name="MCP LiteLLM Overhead Duration (ms)",
                    response_time=duration_ms,
                    response_length=0,
                )
            except (ValueError, TypeError):
                pass

class MCPLoadTestUser(HttpUser):
    """Load test user that performs MCP operations"""
    wait_time = between(0.5, 1)  # Random wait time between requests

    def on_start(self):
        """Set up authentication headers"""
        self.api_key = os.getenv("LITELLM_API_KEY", "sk-1234")
        self.client.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        })
        # Cache available tools to avoid repeated list calls
        self.available_tools = []
        self._refresh_tools()

    def _refresh_tools(self):
        """Fetch available MCP tools from the server"""
        try:
            response = self.client.get(
                "/mcp-rest/tools/list",
                name="List MCP Tools (setup)"
            )
            if response.status_code == 200:
                data = response.json()
                self.available_tools = data.get("tools", [])
                if self.available_tools:
                    print(f"[User {self.environment.runner.user_count}] Found {len(self.available_tools)} MCP tools")
        except Exception as e:
            print(f"Error refreshing tools: {e}")

    @task(3)
    def list_mcp_tools(self):
        """
        List all available MCP tools.
        This is a lightweight operation that tests the discovery endpoint.
        """
        response = self.client.get(
            "/mcp-rest/tools/list",
            name="List MCP Tools"
        )
        
        if response.status_code != 200:
            with open("mcp_errors.txt", "a") as error_log:
                error_log.write(f"List tools error: {response.status_code} - {response.text}\n")

    @task(1)
    def call_mcp_tool(self):
        """
        Call an MCP tool if available.
        Adjust the tool name and arguments based on your MCP server configuration.
        """
        if not self.available_tools:
            # Refresh tools if we don't have any cached
            self._refresh_tools()
            return
        
        # Use the first available tool for testing
        # In a real scenario, you might want to cycle through different tools
        tool = self.available_tools[0]
        tool_name = tool["name"]
        
        # Extract required parameters from tool schema
        required_params = tool.get("inputSchema", {}).get("required", [])
        properties = tool.get("inputSchema", {}).get("properties", {})
        
        # Build arguments dynamically based on tool schema
        arguments = {}
        for param in required_params:
            param_schema = properties.get(param, {})
            param_type = param_schema.get("type", "string")
            
            # Generate test values based on parameter type
            if param_type == "string":
                arguments[param] = f"test-{uuid.uuid4()}"
            elif param_type == "number" or param_type == "integer":
                arguments[param] = 123
            elif param_type == "boolean":
                arguments[param] = True
            else:
                arguments[param] = f"test-{uuid.uuid4()}"
        
        # If no required params, add at least one optional param if available
        if not arguments and properties:
            first_param = list(properties.keys())[0]
            arguments[first_param] = f"test-{uuid.uuid4()}"
        
        payload = {
            "tool_name": tool_name,
            "arguments": arguments
        }
        
        response = self.client.post(
            "/mcp-rest/tools/call",
            json=payload,
            name=f"Call MCP Tool ({tool_name})"
        )
        
        if response.status_code != 200:
            with open("mcp_errors.txt", "a") as error_log:
                error_log.write(
                    f"Call tool error ({tool_name}): {response.status_code} - {response.text}\n"
                )

    @task(1)
    def test_mcp_via_responses_api(self):
        """
        Test MCP tools via the Responses API (more realistic usage pattern).
        This simulates how clients actually use MCP with LLMs.
        """
        payload = {
            "model": os.getenv("TEST_MODEL", "gpt-4o"),  # Use your configured model
            "input": [
                {
                    "role": "user",
                    "content": f"Test MCP integration {uuid.uuid4()}. Please list available tools.",
                    "type": "message"
                }
            ],
            "tools": [
                {
                    "type": "mcp",
                    "server_label": "litellm",
                    "server_url": "litellm_proxy",
                    "require_approval": "never"
                }
            ],
            "tool_choice": "auto"  # Let the model decide whether to use tools
        }
        
        response = self.client.post(
            "/v1/responses",
            json=payload,
            name="MCP via Responses API"
        )
        
        if response.status_code != 200:
            with open("mcp_errors.txt", "a") as error_log:
                error_log.write(f"Responses API error: {response.status_code} - {response.text}\n")

