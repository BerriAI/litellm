import React from "react";
import {
  Card,
  Typography,
  Space,
  Alert,
  Button,
  Divider,
  Tabs,
} from "antd";
import { CopyIcon, Code, Terminal, Globe } from "lucide-react";
import { getProxyBaseUrl } from "../networking";

const { Title, Text, Paragraph } = Typography;

const MCPConnect: React.FC = () => {
  const proxyBaseUrl = getProxyBaseUrl();

  const OpenAITab = () => (
    <Space direction="vertical" size="middle" className="w-full">
      <Paragraph>
        Connect OpenAI's Responses API to your LiteLLM MCP server
      </Paragraph>
      
      <div>
        <Text strong>Get an OpenAI API Key</Text>
        <Paragraph>
          First, make sure that you have a valid OpenAI API key.
          To view or generate a new OpenAI API Key, go to the{" "}
          <a href="https://platform.openai.com/api-keys" target="_blank" rel="noopener noreferrer">
            OpenAI platform API Keys page
          </a>
        </Paragraph>
        <Paragraph>Export your API key as an environment variable:</Paragraph>
        <Card className="bg-gray-50">
          <Text code>export OPENAI_API_KEY="sk-..."</Text>
        </Card>
      </div>

      <div>
        <Text strong>MCP server information</Text>
        <div className="mt-2">
          <Text strong>MCP Server URL</Text>
          <Paragraph>The URL for this MCP server.</Paragraph>
          <Card className="bg-gray-50">
            <Text code className="text-sm">
              {proxyBaseUrl}/mcp
            </Text>
          </Card>
        </div>
      </div>

      <div>
        <Text strong>Using OpenAI's Responses API</Text>
        <Paragraph>
          OpenAI's{" "}
          <a href="https://platform.openai.com/docs/api-reference/responses" target="_blank" rel="noopener noreferrer">
            Responses API
          </a>{" "}
          can be used to call LiteLLM MCP from anywhere.
        </Paragraph>
        
        <Text strong>Code Example</Text>
        <Card className="bg-gray-50 mt-2">
          <pre className="text-xs overflow-x-auto">
{`curl --location 'https://api.openai.com/v1/responses' \\
--header 'Content-Type: application/json' \\
--header "Authorization: Bearer $OPENAI_API_KEY" \\
--data '{
    "model": "gpt-4.1",
    "tools": [
        {
            "type": "mcp",
            "server_label": "litellm",
            "server_url": "${proxyBaseUrl}/mcp",
            "require_approval": "never",
            "headers": {
                "Authorization": "Bearer YOUR_LITELLM_API_KEY"
            }
        }
    ],
    "input": "Run available tools",
    "tool_choice": "required"
}'`}
          </pre>
        </Card>
      </div>
    </Space>
  );

  const CursorTab = () => (
    <Space direction="vertical" size="middle" className="w-full">
      <Paragraph>
        Use tools directly from Cursor IDE with LiteLLM MCP. Enable your AI assistant to perform real-world tasks through a simple, secure connection without leaving your coding environment.
      </Paragraph>
      
      <div>
        <Text strong>Configuring LiteLLM MCP in Cursor</Text>
        <Paragraph>
          Follow these steps to configure LiteLLM MCP in your Cursor IDE:
        </Paragraph>
        
        <ol className="ml-4">
          <li className="mb-2">
            <Text>Open Cursor settings (⇧+⌘+J)</Text>
          </li>
          <li className="mb-2">
            <Text>Navigate to the "MCP Tools" tab and click "New MCP Server"</Text>
          </li>
          <li className="mb-2">
            <Text>Copy/paste the following JSON configuration from below, then hit CMD+S or CTRL+S to save.</Text>
          </li>
          <li className="mb-2">
            <Text>To use LiteLLM MCP within Cursor, set the chat to Agent mode</Text>
          </li>
        </ol>
      </div>

      <Alert
        message="Caution: Treat your MCP server URL like a password! It can be used to run tools attached to this server and access your data."
        type="warning"
        showIcon
      />

      <div>
        <Text strong>Configuration JSON</Text>
        <Card className="bg-gray-50 mt-2">
          <pre className="text-sm">
{`{
  "mcpServers": {
    "LiteLLM": {
      "url": "${proxyBaseUrl}/mcp"
    }
  }
}`}
          </pre>
        </Card>
      </div>
    </Space>
  );

  const StreamableHTTPTab = () => (
    <Space direction="vertical" size="middle" className="w-full">
      <Paragraph>
        Each MCP client supports different transports. See the documentation for your client to know which transport to use.{" "}
        <a href="#" target="_blank" rel="noopener noreferrer">
          Learn more about MCP transports here
        </a>
      </Paragraph>
      
      <div>
        <Text strong>Connect with server-specific URL</Text>
        <Paragraph>
          Use this URL to connect to LiteLLM MCP with any MCP client. This URL is specific to this server.
        </Paragraph>
      </div>

      <Alert
        message="Caution: Treat your MCP server URL like a password! It can be used to run tools attached to this server and access your data."
        type="warning"
        showIcon
      />

      <div>
        <Text strong>Server URL</Text>
        <Card className="mt-2 bg-gray-50">
          <Text code className="text-sm">
            {proxyBaseUrl}/mcp
          </Text>
        </Card>
      </div>
    </Space>
  );

  const tabItems = [
    {
      key: "openai",
      label: (
        <span className="flex items-center gap-2">
          <Code size={16} />
          OpenAI API
        </span>
      ),
      children: <OpenAITab />,
    },
    {
      key: "cursor",
      label: (
        <span className="flex items-center gap-2">
          <Terminal size={16} />
          Cursor
        </span>
      ),
      children: <CursorTab />,
    },
    {
      key: "streamable-http",
      label: (
        <span className="flex items-center gap-2">
          <Globe size={16} />
          Streamable HTTP
        </span>
      ),
      children: <StreamableHTTPTab />,
    },
  ];

  return (
    <div className="p-6">
      <Space direction="vertical" size="large" className="w-full">
        <div>
          <Title level={2}>Connect to your MCP client</Title>
          <Paragraph>
            Use tools directly from any MCP client with LiteLLM MCP. Enable your AI assistant to perform real-world tasks through a simple, secure connection.
          </Paragraph>
        </div>

        <Tabs
          defaultActiveKey="openai"
          items={tabItems}
          className="w-full"
        />
      </Space>
    </div>
  );
};

export default MCPConnect; 