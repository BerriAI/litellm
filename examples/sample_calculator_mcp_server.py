#!/usr/bin/env python3
"""
Sample Calculator MCP Server

This is a simple example MCP server that provides basic calculator functionality.
It demonstrates how to create a custom MCP server that can be integrated as a builtin tool.

Usage:
    python examples/sample_calculator_mcp_server.py

Tools provided:
- add: Add two numbers
- subtract: Subtract two numbers
- multiply: Multiply two numbers
- divide: Divide two numbers (with zero division protection)
- calculate: Evaluate a mathematical expression safely
"""

import asyncio
import json
import sys
from typing import Any, Dict

# MCP Protocol Messages
class MCPMessage:
    def __init__(self, id: str, method: str, params: Dict[str, Any] = None):
        self.id = id
        self.method = method
        self.params = params or {}

class MCPResponse:
    def __init__(self, id: str, result: Any = None, error: Dict[str, Any] = None):
        self.id = id
        self.result = result
        self.error = error

class CalculatorMCPServer:
    """Simple Calculator MCP Server"""

    def __init__(self):
        self.tools = [
            {
                "name": "add",
                "description": "Add two numbers together",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number", "description": "First number"},
                        "b": {"type": "number", "description": "Second number"}
                    },
                    "required": ["a", "b"]
                }
            },
            {
                "name": "subtract",
                "description": "Subtract the second number from the first",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number", "description": "First number (minuend)"},
                        "b": {"type": "number", "description": "Second number (subtrahend)"}
                    },
                    "required": ["a", "b"]
                }
            },
            {
                "name": "multiply",
                "description": "Multiply two numbers",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number", "description": "First number"},
                        "b": {"type": "number", "description": "Second number"}
                    },
                    "required": ["a", "b"]
                }
            },
            {
                "name": "divide",
                "description": "Divide the first number by the second",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number", "description": "Dividend"},
                        "b": {"type": "number", "description": "Divisor"}
                    },
                    "required": ["a", "b"]
                }
            },
            {
                "name": "calculate",
                "description": "Evaluate a mathematical expression safely",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Mathematical expression to evaluate (e.g., '2 + 3 * 4')"
                        }
                    },
                    "required": ["expression"]
                }
            }
        ]

    async def handle_initialize(self, message: MCPMessage) -> MCPResponse:
        """Handle initialization request"""
        return MCPResponse(
            id=message.id,
            result={
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "calculator-mcp-server",
                    "version": "1.0.0",
                    "description": "Simple calculator MCP server for demonstration"
                }
            }
        )

    async def handle_list_tools(self, message: MCPMessage) -> MCPResponse:
        """Handle tools/list request"""
        return MCPResponse(
            id=message.id,
            result={"tools": self.tools}
        )

    async def handle_call_tool(self, message: MCPMessage) -> MCPResponse:
        """Handle tools/call request"""
        tool_name = message.params.get("name")
        arguments = message.params.get("arguments", {})

        try:
            if tool_name == "add":
                result = self._add(arguments)
            elif tool_name == "subtract":
                result = self._subtract(arguments)
            elif tool_name == "multiply":
                result = self._multiply(arguments)
            elif tool_name == "divide":
                result = self._divide(arguments)
            elif tool_name == "calculate":
                result = self._calculate(arguments)
            else:
                return MCPResponse(
                    id=message.id,
                    error={
                        "code": -32601,
                        "message": f"Unknown tool: {tool_name}"
                    }
                )

            return MCPResponse(
                id=message.id,
                result={
                    "content": [
                        {
                            "type": "text",
                            "text": str(result)
                        }
                    ]
                }
            )

        except Exception as e:
            return MCPResponse(
                id=message.id,
                error={
                    "code": -32603,
                    "message": f"Tool execution error: {str(e)}"
                }
            )

    def _add(self, args: Dict[str, Any]) -> float:
        """Add two numbers"""
        a = float(args["a"])
        b = float(args["b"])
        result = a + b
        return f"{a} + {b} = {result}"

    def _subtract(self, args: Dict[str, Any]) -> str:
        """Subtract two numbers"""
        a = float(args["a"])
        b = float(args["b"])
        result = a - b
        return f"{a} - {b} = {result}"

    def _multiply(self, args: Dict[str, Any]) -> str:
        """Multiply two numbers"""
        a = float(args["a"])
        b = float(args["b"])
        result = a * b
        return f"{a} × {b} = {result}"

    def _divide(self, args: Dict[str, Any]) -> str:
        """Divide two numbers"""
        a = float(args["a"])
        b = float(args["b"])

        if b == 0:
            raise ValueError("Cannot divide by zero")

        result = a / b
        return f"{a} ÷ {b} = {result}"

    def _calculate(self, args: Dict[str, Any]) -> str:
        """Safely evaluate a mathematical expression"""
        expression = args["expression"].strip()

        # Simple safety check - only allow basic math operations and numbers
        allowed_chars = set("0123456789+-*/(). ")
        if not all(c in allowed_chars for c in expression):
            raise ValueError("Expression contains invalid characters")

        try:
            # Use eval with restricted built-ins for safety
            result = eval(expression, {"__builtins__": {}}, {})
            return f"{expression} = {result}"
        except Exception as e:
            raise ValueError(f"Invalid expression: {str(e)}")

    async def handle_message(self, message_data: str) -> str:
        """Handle incoming MCP message"""
        try:
            data = json.loads(message_data)
            message = MCPMessage(
                id=data.get("id", ""),
                method=data.get("method", ""),
                params=data.get("params", {})
            )

            if message.method == "initialize":
                response = await self.handle_initialize(message)
            elif message.method == "tools/list":
                response = await self.handle_list_tools(message)
            elif message.method == "tools/call":
                response = await self.handle_call_tool(message)
            else:
                response = MCPResponse(
                    id=message.id,
                    error={
                        "code": -32601,
                        "message": f"Method not found: {message.method}"
                    }
                )

            # Format response
            response_data = {"id": response.id}
            if response.result is not None:
                response_data["result"] = response.result
            if response.error is not None:
                response_data["error"] = response.error

            return json.dumps(response_data)

        except Exception as e:
            error_response = {
                "id": "",
                "error": {
                    "code": -32700,
                    "message": f"Parse error: {str(e)}"
                }
            }
            return json.dumps(error_response)


async def run_stdio_server():
    """Run the MCP server using stdio transport"""
    server = CalculatorMCPServer()

    # Calculator MCP Server started (stdio mode)
    # Ready to accept requests...

    while True:
        try:
            # Read JSON-RPC message from stdin
            line = sys.stdin.readline()
            if not line:
                break

            # Process message
            response = await server.handle_message(line.strip())

            # Send response to stdout
            print(response)
            sys.stdout.flush()

        except EOFError:
            break
        except Exception:
            pass  # Error processing message - logging could be added here


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Test mode - demonstrate the server functionality
        async def test_server():
            server = CalculatorMCPServer()

            print("🧮 Testing Calculator MCP Server...\n")

            # Test initialization
            init_msg = '{"id": "1", "method": "initialize", "params": {}}'
            response = await server.handle_message(init_msg)
            print(f"Initialize: {response}\n")

            # Test list tools
            list_msg = '{"id": "2", "method": "tools/list", "params": {}}'
            response = await server.handle_message(list_msg)
            print(f"List tools: {response}\n")

            # Test tool calls
            test_calls = [
                ('{"id": "3", "method": "tools/call", "params": {"name": "add", "arguments": {"a": 5, "b": 3}}}', "Addition"),
                ('{"id": "4", "method": "tools/call", "params": {"name": "multiply", "arguments": {"a": 4, "b": 7}}}', "Multiplication"),
                ('{"id": "5", "method": "tools/call", "params": {"name": "calculate", "arguments": {"expression": "2 + 3 * 4"}}}', "Expression")
            ]

            for call_msg, description in test_calls:
                response = await server.handle_message(call_msg)
                print(f"{description}: {response}\n")

            print("✅ Test completed!")

        asyncio.run(test_server())
    else:
        # Normal stdio mode
        asyncio.run(run_stdio_server())

