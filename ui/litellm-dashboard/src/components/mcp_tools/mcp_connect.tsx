/* eslint-disable react/no-unescaped-entities */

import React, { useState } from "react";
import {
  Card,
  Typography,
  Space,
  Alert,
  Button,
  message,
} from "antd";
import {
  TabPanel,
  TabPanels,
  TabGroup,
  TabList,
  Tab,
  Title as TremorTitle,
  Text as TremorText,
} from "@tremor/react";
import { 
  CopyIcon, 
  Code, 
  Terminal, 
  Globe, 
  CheckIcon, 
  ExternalLinkIcon,
  ShieldAlertIcon,
  KeyIcon,
  ServerIcon,
  Zap
} from "lucide-react";
import { getProxyBaseUrl } from "../networking";

const { Title, Text } = Typography;

const MCPConnect: React.FC = () => {
  const proxyBaseUrl = getProxyBaseUrl();
  const [copiedStates, setCopiedStates] = useState<Record<string, boolean>>({});

  const copyToClipboard = async (text: string, key: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedStates(prev => ({ ...prev, [key]: true }));
      message.success('Copied to clipboard');
      setTimeout(() => {
        setCopiedStates(prev => ({ ...prev, [key]: false }));
      }, 2000);
    } catch (err) {
      message.error('Failed to copy to clipboard');
    }
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
          <Text strong className="text-gray-700">{title}</Text>
        </div>
      )}
      <Card className={`bg-gray-50 border border-gray-200 relative ${className}`}>
        <Button
          type="text"
          size="small"
          icon={copiedStates[copyKey] ? <CheckIcon size={14} /> : <CopyIcon size={14} />}
          onClick={() => copyToClipboard(code, copyKey)}
          className={`absolute top-2 right-2 z-10 transition-all duration-200 ${
            copiedStates[copyKey] 
              ? 'text-green-600 bg-green-50 border-green-200' 
              : 'text-gray-500 hover:text-gray-700 hover:bg-gray-100'
          }`}
        />
        <pre className="text-sm overflow-x-auto pr-10 text-gray-800 font-mono leading-relaxed">
          {code}
        </pre>
      </Card>
    </div>
  );

  const FeatureCard: React.FC<{
    icon: React.ReactNode;
    title: string;
    description: string;
    children: React.ReactNode;
  }> = ({ icon, title, description, children }) => (
    <Card className="border border-gray-200 shadow-sm hover:shadow-md transition-shadow duration-200">
      <div className="flex items-start gap-3 mb-4">
        <div className="flex-shrink-0 w-8 h-8 bg-blue-50 rounded-lg flex items-center justify-center">
          {icon}
        </div>
        <div className="flex-1">
          <Title level={5} className="mb-1 text-gray-800">{title}</Title>
          <Text className="text-gray-600">{description}</Text>
        </div>
      </div>
      {children}
    </Card>
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
        <Text strong className="text-gray-800 block mb-2">{title}</Text>
        {children}
      </div>
    </div>
  );

  const LiteLLMProxyTab = () => (
    <Space direction="vertical" size="large" className="w-full">
      <div className="bg-gradient-to-r from-emerald-50 to-green-50 p-6 rounded-lg border border-emerald-100">
        <div className="flex items-center gap-3 mb-3">
          <Zap className="text-emerald-600" size={24} />
          <Title level={4} className="mb-0 text-emerald-900">LiteLLM Proxy API Integration</Title>
        </div>
        <Text className="text-emerald-700">
          Connect to LiteLLM Proxy Responses API for seamless tool integration with multiple model providers
        </Text>
      </div>
      
      <Space direction="vertical" size="large" className="w-full">
        <FeatureCard
          icon={<KeyIcon className="text-emerald-600" size={16} />}
          title="API Key Setup"
          description="Configure your LiteLLM Proxy API key for authentication"
        >
          <Space direction="vertical" size="middle" className="w-full">
            <div>
              <Text>Get your API key from your LiteLLM Proxy dashboard or contact your administrator</Text>
            </div>
            <CodeBlock
              title="Environment Variable"
              code='export LITELLM_API_KEY="sk-..."'
              copyKey="litellm-env"
            />
          </Space>
        </FeatureCard>

        <FeatureCard
          icon={<ServerIcon className="text-emerald-600" size={16} />}
          title="MCP Server Information"
          description="Connection details for your LiteLLM MCP server"
        >
          <CodeBlock
            title="Server URL"
            code={`${proxyBaseUrl}/mcp`}
            copyKey="litellm-server-url"
          />
        </FeatureCard>

        <FeatureCard
          icon={<Code className="text-emerald-600" size={16} />}
          title="Implementation Example"
          description="Complete cURL example for using the LiteLLM Proxy Responses API"
        >
          <CodeBlock
            code={`curl --location '${proxyBaseUrl}/v1/responses' \\
--header 'Content-Type: application/json' \\
--header "Authorization: Bearer $LITELLM_API_KEY" \\
--data '{
    "model": "gpt-4",
    "tools": [
        {
            "type": "mcp",
            "server_label": "litellm",
            "server_url": "${proxyBaseUrl}/mcp",
            "require_approval": "never",
            "headers": {
                "x-litellm-api-key": "Bearer YOUR_LITELLM_API_KEY"
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
          <Title level={4} className="mb-0 text-blue-900">OpenAI Responses API Integration</Title>
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
            <Text>Get your API key from the{" "}
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
            <CodeBlock
              title="Environment Variable"
              code='export OPENAI_API_KEY="sk-..."'
              copyKey="openai-env"
            />
          </Space>
        </FeatureCard>

        <FeatureCard
          icon={<ServerIcon className="text-blue-600" size={16} />}
          title="MCP Server Information"
          description="Connection details for your LiteLLM MCP server"
        >
          <CodeBlock
            title="Server URL"
            code={`${proxyBaseUrl}/mcp`}
            copyKey="openai-server-url"
          />
        </FeatureCard>

        <FeatureCard
          icon={<Code className="text-blue-600" size={16} />}
          title="Implementation Example"
          description="Complete cURL example for using the Responses API"
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
                "x-litellm-api-key": "Bearer YOUR_LITELLM_API_KEY"
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
          <Title level={4} className="mb-0 text-purple-900">Cursor IDE Integration</Title>
        </div>
        <Text className="text-purple-700">
          Use tools directly from Cursor IDE with LiteLLM MCP. Enable your AI assistant to perform real-world tasks without leaving your coding environment.
        </Text>
      </div>

      <Card className="border border-gray-200">
        <Title level={5} className="mb-4 text-gray-800">Setup Instructions</Title>
        <Space direction="vertical" size="large" className="w-full">
          <StepCard
            step={1}
            title="Open Cursor Settings"
          >
            <Text className="text-gray-600">Use the keyboard shortcut <code className="bg-gray-100 px-2 py-1 rounded">⇧+⌘+J</code> (Mac) or <code className="bg-gray-100 px-2 py-1 rounded">Ctrl+Shift+J</code> (Windows/Linux)</Text>
          </StepCard>

          <StepCard
            step={2}
            title="Navigate to MCP Tools"
          >
            <Text className="text-gray-600">Go to the "MCP Tools" tab and click "New MCP Server"</Text>
          </StepCard>

          <StepCard
            step={3}
            title="Add Configuration"
          >
            <Text className="text-gray-600 mb-3">Copy the JSON configuration below and paste it into Cursor, then save with <code className="bg-gray-100 px-2 py-1 rounded">Cmd+S</code> or <code className="bg-gray-100 px-2 py-1 rounded">Ctrl+S</code></Text>
            <CodeBlock
              code={`{
  "mcpServers": {
    "LiteLLM": {
      "url": "${proxyBaseUrl}/mcp",
      "headers": {
        "x-litellm-api-key": "Bearer $LITELLM_API_KEY"
      }
    }
  }
}`}
copyKey="cursor-config"
            />
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
          <Title level={4} className="mb-0 text-green-900">Streamable HTTP Transport</Title>
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
            <Text>Each MCP client supports different transports. Refer to your client documentation to determine the appropriate transport method.</Text>
          </div>
          <CodeBlock
            title="Server URL"
            code={`${proxyBaseUrl}/mcp`}
            copyKey="http-server-url"
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
          <TremorTitle className="text-3xl font-bold text-gray-900 mb-3">
            Connect to your MCP client
          </TremorTitle>
          <TremorText className="text-lg text-gray-600">
            Use tools directly from any MCP client with LiteLLM MCP. Enable your AI assistant to perform real-world tasks through a simple, secure connection.
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