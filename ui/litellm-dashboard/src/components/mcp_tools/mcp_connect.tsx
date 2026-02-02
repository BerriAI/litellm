/* eslint-disable react/no-unescaped-entities */

import React, { useState } from "react";
import { Card, Typography, Space, Alert, Button, Switch, Form, Collapse } from "antd";
import { TabPanel, TabPanels, TabGroup, TabList, Tab, Title as TremorTitle, Text as TremorText } from "@tremor/react";
import { CopyIcon, Code, Terminal, Globe, CheckIcon, ExternalLinkIcon, KeyIcon, ServerIcon, Zap } from "lucide-react";
import { getProxyBaseUrl } from "../networking";
import { copyToClipboard as utilCopyToClipboard } from "../../utils/dataUtils";

const { Title, Text } = Typography;
const { Panel } = Collapse;

interface CodeBlockProps {
  code: string;
  title?: string;
  copyKey: string;
  className?: string;
}

interface FeatureCardProps {
  icon: React.ReactNode;
  title: string;
  description: string;
  children: React.ReactNode;
  serverName?: string;
  accessGroups?: string[];
}

const FeatureCard: React.FC<FeatureCardProps> = ({
  icon,
  title,
  description,
  children,
  serverName,
  accessGroups = ["dev-group"],
}) => {
  const [useServerHeader, setUseServerHeader] = useState(false);

  const getHeadersConfig = () => {
    const headers: Record<string, any> = {
      "x-litellm-api-key": "Bearer YOUR_LITELLM_API_KEY",
    };
    if (useServerHeader && serverName) {
      const formattedServerName = serverName.replace(/\s+/g, "_");
      // Include both server name and access groups in the same header (comma-separated string)
      const serverAndGroups = [formattedServerName, ...accessGroups].join(",");
      headers["x-mcp-servers"] = serverAndGroups;
    }
    return headers;
  };

  return (
    <Card className="border border-gray-200">
      <div className="flex items-center gap-3 mb-3">
        <span className="p-2 rounded-lg bg-gray-50">{icon}</span>
        <div>
          <Title level={5} className="mb-0">
            {title}
          </Title>
          <Text className="text-gray-600">{description}</Text>
        </div>
      </div>
      {serverName && (title === "Implementation Example" || title === "Configuration") && (
        <Form.Item className="mb-4">
          <div className="flex items-center gap-2 mb-2">
            <Switch size="small" checked={useServerHeader} onChange={setUseServerHeader} />
            <Text className="text-sm">
              Limit tools to specific MCP servers or MCP groups by passing the <code>x-mcp-servers</code> header
            </Text>
          </div>
          {useServerHeader && (
            <Alert
              className="mt-2"
              type="info"
              showIcon
              message="Two Options"
              description={
                <div>
                  <p>
                    <strong>Option 1:</strong> Get a specific server: <code>"{serverName.replace(/\s+/g, "_")}"</code>
                  </p>
                  <p>
                    <strong>Option 2:</strong> Get a group of MCPs: <code>"dev-group"</code>
                  </p>
                  <p className="mt-2 text-sm text-gray-600">
                    You can also mix both: <code>"Server1,dev-group"</code>
                  </p>
                </div>
              }
            />
          )}
        </Form.Item>
      )}
      {React.Children.map(children, (child) => {
        if (
          React.isValidElement<CodeBlockProps>(child) &&
          child.props.hasOwnProperty("code") &&
          child.props.hasOwnProperty("copyKey")
        ) {
          const code = child.props.code;
          if (code && code.includes('"headers":')) {
            return React.cloneElement(child, {
              code: code.replace(/"headers":\s*{[^}]*}/, `"headers": ${JSON.stringify(getHeadersConfig(), null, 8)}`),
            });
          }
        }
        return child;
      })}
    </Card>
  );
};

interface MCPConnectProps {
  currentServerAccessGroups?: string[];
}

const MCPConnect: React.FC<MCPConnectProps> = ({ currentServerAccessGroups = [] }) => {
  const proxyBaseUrl = getProxyBaseUrl();
  const [copiedStates, setCopiedStates] = useState<Record<string, boolean>>({});
  const [serverHeaders, setServerHeaders] = useState<Record<string, string[]>>({
    openai: [],
    litellm: [],
    cursor: [],
    http: [],
  });
  const [currentServer] = useState("Zapier_MCP"); // This should match the current server being viewed

  const copyToClipboard = async (text: string, key: string) => {
    const success = await utilCopyToClipboard(text);
    if (success) {
      setCopiedStates((prev) => ({ ...prev, [key]: true }));
      setTimeout(() => {
        setCopiedStates((prev) => ({ ...prev, [key]: false }));
      }, 2000);
    }
  };

  const getHeadersConfig = (type: string) => {
    const headers: Record<string, any> = {
      "x-litellm-api-key": "Bearer YOUR_LITELLM_API_KEY",
    };

    if (serverHeaders[type]?.length > 0) {
      // Format server names (replace spaces with underscores)
      const formattedServers = serverHeaders[type].map((s) => s.replace(/\s+/g, "_"));

      // Use comma-separated string (can include both servers and access groups)
      headers["x-mcp-servers"] = formattedServers.join(",");
    }

    return headers;
  };

  const CodeBlock: React.FC<{
    code: string;
    copyKey: string;
    title?: string;
    className?: string;
  }> = ({ code, copyKey, title, className = "" }) => (
    <div className="relative group">
      {title && (
        <div className="flex items-center gap-2 mb-2">
          <Code size={16} className="text-blue-600" />
          <Text strong className="text-gray-700">
            {title}
          </Text>
        </div>
      )}
      <Card className={`bg-gray-50 border border-gray-200 relative ${className}`}>
        <Button
          type="text"
          size="small"
          icon={copiedStates[copyKey] ? <CheckIcon size={12} /> : <CopyIcon size={12} />}
          onClick={() => copyToClipboard(code, copyKey)}
          className={`absolute top-2 right-2 z-10 transition-all duration-200 ${
            copiedStates[copyKey]
              ? "text-green-600 bg-green-50 border-green-200"
              : "text-gray-500 hover:text-gray-700 hover:bg-gray-100"
          }`}
        />
        <pre className="text-sm overflow-x-auto pr-10 text-gray-800 font-mono leading-relaxed">{code}</pre>
      </Card>
    </div>
  );

  const StepCard: React.FC<{
    step: number;
    title: string;
    children: React.ReactNode;
  }> = ({ step, title, children }) => (
    <div className="flex gap-4">
      <div className="flex-shrink-0">
        <div className="w-8 h-8 bg-blue-600 text-white rounded-full flex items-center justify-center text-sm font-semibold">
          {step}
        </div>
      </div>
      <div className="flex-1">
        <Text strong className="text-gray-800 block mb-2">
          {title}
        </Text>
        {children}
      </div>
    </div>
  );

  const LiteLLMProxyTab = () => (
    <Space direction="vertical" size="large" className="w-full">
      <div className="bg-gradient-to-r from-emerald-50 to-green-50 p-6 rounded-lg border border-emerald-100">
        <div className="flex items-center gap-3 mb-3">
          <Zap className="text-emerald-600" size={24} />
          <Title level={4} className="mb-0 text-emerald-900">
            LiteLLM Proxy API Integration
          </Title>
        </div>
        <Text className="text-emerald-700">
          Connect to LiteLLM Proxy Responses API for seamless tool integration with multiple model providers
        </Text>
      </div>

      <Space direction="vertical" size="large" className="w-full">
        <FeatureCard
          icon={<KeyIcon className="text-emerald-600" size={16} />}
          title="Virtual Key Setup"
          description="Configure your LiteLLM Proxy Virtual Key for authentication"
        >
          <Space direction="vertical" size="middle" className="w-full">
            <div>
              <Text>Get your Virtual Key from your LiteLLM Proxy dashboard or contact your administrator</Text>
            </div>
            <CodeBlock title="Environment Variable" code='export LITELLM_API_KEY="sk-..."' copyKey="litellm-env" />
          </Space>
        </FeatureCard>

        <FeatureCard
          icon={<ServerIcon className="text-emerald-600" size={16} />}
          title="MCP Server Information"
          description="Connection details for your LiteLLM MCP server"
        >
          <CodeBlock title="Server URL" code={`${proxyBaseUrl}/mcp`} copyKey="litellm-server-url" />
        </FeatureCard>

        <FeatureCard
          icon={<Code className="text-emerald-600" size={16} />}
          title="Implementation Example"
          description="Complete cURL example for using the LiteLLM Proxy Responses API"
          serverName={currentServer}
          accessGroups={["dev-group"]}
        >
          <CodeBlock
            code={`curl --location '${proxyBaseUrl}/v1/responses' \\
--header 'Content-Type: application/json' \\
--header "Authorization: Bearer $LITELLM_VIRTUAL_KEY" \\
--data '{
    "model": "gpt-4",
    "tools": [
        {
            "type": "mcp",
            "server_label": "litellm",
            "server_url": "${proxyBaseUrl}/mcp",
            "require_approval": "never",
            "headers": {
                "x-litellm-api-key": "Bearer YOUR_LITELLM_VIRTUAL_KEY",
                "x-mcp-servers": "Zapier_MCP,dev-group"
            }
        }
    ],
    "input": "Run available tools",
    "tool_choice": "required"
}'`}
            copyKey="litellm-curl"
            className="text-xs"
          />
        </FeatureCard>
      </Space>
    </Space>
  );

  const OpenAITab = () => (
    <Space direction="vertical" size="large" className="w-full">
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-6 rounded-lg border border-blue-100">
        <div className="flex items-center gap-3 mb-3">
          <Code className="text-blue-600" size={24} />
          <Title level={4} className="mb-0 text-blue-900">
            OpenAI Responses API Integration
          </Title>
        </div>
        <Text className="text-blue-700">
          Connect OpenAI Responses API to your LiteLLM MCP server for seamless tool integration
        </Text>
      </div>

      <Space direction="vertical" size="large" className="w-full">
        <FeatureCard
          icon={<KeyIcon className="text-blue-600" size={16} />}
          title="API Key Setup"
          description="Configure your OpenAI API key for authentication"
        >
          <Space direction="vertical" size="middle" className="w-full">
            <div>
              {/* eslint-disable-next-line react/no-unescaped-entities */}
              <Text>
                Get your API key from the{" "}
                <a
                  href="https://platform.openai.com/api-keys"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-600 hover:text-blue-700 inline-flex items-center gap-1"
                >
                  OpenAI platform <ExternalLinkIcon size={12} />
                </a>
              </Text>
            </div>
            <CodeBlock title="Environment Variable" code='export OPENAI_API_KEY="sk-..."' copyKey="openai-env" />
          </Space>
        </FeatureCard>

        <FeatureCard
          icon={<ServerIcon className="text-blue-600" size={16} />}
          title="MCP Server Information"
          description="Connection details for your LiteLLM MCP server"
        >
          <CodeBlock title="Server URL" code={`${proxyBaseUrl}/mcp`} copyKey="openai-server-url" />
        </FeatureCard>

        <FeatureCard
          icon={<Code className="text-blue-600" size={16} />}
          title="Implementation Example"
          description="Complete cURL example for using the Responses API"
          serverName="Zapier Gmail"
          accessGroups={["dev-group"]}
        >
          <CodeBlock
            code={`curl --location 'https://api.openai.com/v1/responses' \\
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
                "x-litellm-api-key": "Bearer YOUR_LITELLM_API_KEY",
                "x-mcp-servers": "Zapier_MCP,dev-group"
            }
        }
    ],
    "input": "Run available tools",
    "tool_choice": "required"
}'`}
            copyKey="openai-curl"
            className="text-xs"
          />
        </FeatureCard>
      </Space>
    </Space>
  );

  const CursorTab = () => (
    <Space direction="vertical" size="large" className="w-full">
      <div className="bg-gradient-to-r from-purple-50 to-blue-50 p-6 rounded-lg border border-purple-100">
        <div className="flex items-center gap-3 mb-3">
          <Terminal className="text-purple-600" size={24} />
          <Title level={4} className="mb-0 text-purple-900">
            Cursor IDE Integration
          </Title>
        </div>
        <Text className="text-purple-700">
          Use tools directly from Cursor IDE with LiteLLM MCP. Enable your AI assistant to perform real-world tasks
          without leaving your coding environment.
        </Text>
      </div>

      <Card className="border border-gray-200">
        <Title level={5} className="mb-4 text-gray-800">
          Setup Instructions
        </Title>
        <Space direction="vertical" size="large" className="w-full">
          <StepCard step={1} title="Open Cursor Settings">
            <Text className="text-gray-600">
              Use the keyboard shortcut <code className="bg-gray-100 px-2 py-1 rounded">⇧+⌘+J</code> (Mac) or{" "}
              <code className="bg-gray-100 px-2 py-1 rounded">Ctrl+Shift+J</code> (Windows/Linux)
            </Text>
          </StepCard>

          <StepCard step={2} title="Navigate to MCP Tools">
            <Text className="text-gray-600">Go to the "MCP Tools" tab and click "New MCP Server"</Text>
          </StepCard>

          <StepCard step={3} title="Add Configuration">
            <Text className="text-gray-600 mb-3">
              Copy the JSON configuration below and paste it into Cursor, then save with{" "}
              <code className="bg-gray-100 px-2 py-1 rounded">Cmd+S</code> or{" "}
              <code className="bg-gray-100 px-2 py-1 rounded">Ctrl+S</code>
            </Text>
            <FeatureCard
              icon={<Code className="text-purple-600" size={16} />}
              title="Configuration"
              description="Cursor MCP configuration"
              serverName="Zapier Gmail"
              accessGroups={["dev-group"]}
            >
              <CodeBlock
                code={`{
  "mcpServers": {
    "Zapier_MCP": {
      "url": "${proxyBaseUrl}/mcp",
      "headers": {
        "x-litellm-api-key": "Bearer YOUR_LITELLM_API_KEY",
        "x-mcp-servers": "Zapier_MCP,dev-group"
      }
    }
  }
}`}
                copyKey="cursor-config"
                className="text-xs"
              />
            </FeatureCard>
          </StepCard>
        </Space>
      </Card>
    </Space>
  );

  const StreamableHTTPTab = () => (
    <Space direction="vertical" size="large" className="w-full">
      <div className="bg-gradient-to-r from-green-50 to-teal-50 p-6 rounded-lg border border-green-100">
        <div className="flex items-center gap-3 mb-3">
          <Globe className="text-green-600" size={24} />
          <Title level={4} className="mb-0 text-green-900">
            Streamable HTTP Transport
          </Title>
        </div>
        <Text className="text-green-700">
          Connect to LiteLLM MCP using HTTP transport. Compatible with any MCP client that supports HTTP streaming.
        </Text>
      </div>

      <FeatureCard
        icon={<Globe className="text-green-600" size={16} />}
        title="Universal MCP Connection"
        description="Use this URL with any MCP client that supports HTTP transport"
      >
        <Space direction="vertical" size="middle" className="w-full">
          <div>
            <Text>
              Each MCP client supports different transports. Refer to your client documentation to determine the
              appropriate transport method.
            </Text>
          </div>
          <CodeBlock title="Server URL" code={`${proxyBaseUrl}/mcp`} copyKey="http-server-url" />
          <CodeBlock
            title="Headers Configuration"
            code={JSON.stringify(
              {
                "x-litellm-api-key": "Bearer YOUR_LITELLM_API_KEY",
              },
              null,
              2,
            )}
            copyKey="http-headers"
          />
          <div className="mt-4">
            <Button
              type="link"
              className="p-0 h-auto text-blue-600 hover:text-blue-700"
              href="https://modelcontextprotocol.io/docs/concepts/transports"
              icon={<ExternalLinkIcon size={14} />}
            >
              Learn more about MCP transports
            </Button>
          </div>
        </Space>
      </FeatureCard>
    </Space>
  );

  return (
    <div>
      <Space direction="vertical" size="large" className="w-full">
        <div>
          <TremorTitle className="text-3xl font-bold text-gray-900 mb-3">Connect to your MCP client</TremorTitle>
          <TremorText className="text-lg text-gray-600">
            Use tools directly from any MCP client with LiteLLM MCP. Enable your AI assistant to perform real-world
            tasks through a simple, secure connection.
          </TremorText>
        </div>

        <TabGroup className="w-full">
          <TabList className="flex justify-start mt-8 mb-6">
            <div className="flex bg-gray-100 p-1 rounded-lg">
              <Tab className="px-6 py-3 rounded-md transition-all duration-200">
                <span className="flex items-center gap-2 font-medium">
                  <Code size={18} />
                  OpenAI API
                </span>
              </Tab>
              <Tab className="px-6 py-3 rounded-md transition-all duration-200">
                <span className="flex items-center gap-2 font-medium">
                  <Zap size={18} />
                  LiteLLM Proxy
                </span>
              </Tab>
              <Tab className="px-6 py-3 rounded-md transition-all duration-200">
                <span className="flex items-center gap-2 font-medium">
                  <Terminal size={18} />
                  Cursor
                </span>
              </Tab>
              <Tab className="px-6 py-3 rounded-md transition-all duration-200">
                <span className="flex items-center gap-2 font-medium">
                  <Globe size={18} />
                  Streamable HTTP
                </span>
              </Tab>
            </div>
          </TabList>
          <TabPanels>
            <TabPanel className="mt-6">
              <OpenAITab />
            </TabPanel>
            <TabPanel className="mt-6">
              <LiteLLMProxyTab />
            </TabPanel>
            <TabPanel className="mt-6">
              <CursorTab />
            </TabPanel>
            <TabPanel className="mt-6">
              <StreamableHTTPTab />
            </TabPanel>
          </TabPanels>
        </TabGroup>
      </Space>
    </div>
  );
};

export default MCPConnect;
