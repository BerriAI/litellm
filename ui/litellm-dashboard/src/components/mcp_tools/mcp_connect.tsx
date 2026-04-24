/* eslint-disable react/no-unescaped-entities */

import React, { useState } from "react";
import { Card } from "@/components/ui/card";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  CopyIcon,
  Code,
  Terminal,
  Globe,
  CheckIcon,
  ExternalLinkIcon,
  KeyIcon,
  ServerIcon,
  Zap,
  Info,
} from "lucide-react";
import { getProxyBaseUrl } from "../networking";
import { copyToClipboard as utilCopyToClipboard } from "../../utils/dataUtils";

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
      const serverAndGroups = [formattedServerName, ...accessGroups].join(",");
      headers["x-mcp-servers"] = serverAndGroups;
    }
    return headers;
  };

  const featureCardSwitchId = React.useId();

  return (
    <Card className="border border-border p-6">
      <div className="flex items-center gap-3 mb-3">
        <span className="p-2 rounded-lg bg-muted">{icon}</span>
        <div>
          <h5 className="text-base font-semibold mb-0">{title}</h5>
          <p className="text-muted-foreground text-sm m-0">{description}</p>
        </div>
      </div>
      {serverName && (title === "Implementation Example" || title === "Configuration") && (
        <div className="mb-4">
          <div className="flex items-center gap-2 mb-2">
            <Switch
              id={featureCardSwitchId}
              checked={useServerHeader}
              onCheckedChange={setUseServerHeader}
            />
            <Label htmlFor={featureCardSwitchId} className="text-sm font-normal">
              Limit tools to specific MCP servers or MCP groups by passing the{" "}
              <code>x-mcp-servers</code> header
            </Label>
          </div>
          {useServerHeader && (
            <Alert className="mt-2">
              <Info className="h-4 w-4" />
              <AlertTitle>Two Options</AlertTitle>
              <AlertDescription>
                <p>
                  <strong>Option 1:</strong> Get a specific server:{" "}
                  <code>"{serverName.replace(/\s+/g, "_")}"</code>
                </p>
                <p>
                  <strong>Option 2:</strong> Get a group of MCPs: <code>"dev-group"</code>
                </p>
                <p className="mt-2 text-sm text-muted-foreground">
                  You can also mix both: <code>"Server1,dev-group"</code>
                </p>
              </AlertDescription>
            </Alert>
          )}
        </div>
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
              code: code.replace(
                /"headers":\s*{[^}]*}/,
                `"headers": ${JSON.stringify(getHeadersConfig(), null, 8)}`,
              ),
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
  const [currentServer] = useState("Zapier_MCP");

  const copyToClipboard = async (text: string, key: string) => {
    const success = await utilCopyToClipboard(text);
    if (success) {
      setCopiedStates((prev) => ({ ...prev, [key]: true }));
      setTimeout(() => {
        setCopiedStates((prev) => ({ ...prev, [key]: false }));
      }, 2000);
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
          <Code size={16} className="text-primary" />
          <span className="font-semibold text-foreground">{title}</span>
        </div>
      )}
      <Card className={`bg-muted border border-border relative p-4 ${className}`}>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => copyToClipboard(code, copyKey)}
          className={`absolute top-2 right-2 z-10 transition-all duration-200 ${
            copiedStates[copyKey] ? "text-primary" : "text-muted-foreground hover:text-foreground"
          }`}
        >
          {copiedStates[copyKey] ? <CheckIcon size={12} /> : <CopyIcon size={12} />}
        </Button>
        <pre className="text-sm overflow-x-auto pr-10 text-foreground font-mono leading-relaxed">
          {code}
        </pre>
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
        <div className="w-8 h-8 bg-primary text-primary-foreground rounded-full flex items-center justify-center text-sm font-semibold">
          {step}
        </div>
      </div>
      <div className="flex-1">
        <span className="font-semibold text-foreground block mb-2">{title}</span>
        {children}
      </div>
    </div>
  );

  const LiteLLMProxyTab = () => (
    <div className="flex flex-col gap-6 w-full">
      <div className="bg-gradient-to-r from-emerald-50 to-green-50 dark:from-emerald-950/30 dark:to-green-950/30 p-6 rounded-lg border border-emerald-100 dark:border-emerald-900">
        <div className="flex items-center gap-3 mb-3">
          <Zap className="text-emerald-600 dark:text-emerald-400" size={24} />
          <h4 className="mb-0 text-lg font-semibold text-emerald-900 dark:text-emerald-100">
            LiteLLM Proxy API Integration
          </h4>
        </div>
        <p className="text-emerald-700 dark:text-emerald-200">
          Connect to LiteLLM Proxy Responses API for seamless tool integration with multiple model providers
        </p>
      </div>

      <div className="flex flex-col gap-6 w-full">
        <FeatureCard
          icon={<KeyIcon className="text-emerald-600 dark:text-emerald-400" size={16} />}
          title="Virtual Key Setup"
          description="Configure your LiteLLM Proxy Virtual Key for authentication"
        >
          <div className="flex flex-col gap-4 w-full">
            <div>
              <span className="text-foreground">
                Get your Virtual Key from your LiteLLM Proxy dashboard or contact your administrator
              </span>
            </div>
            <CodeBlock
              title="Environment Variable"
              code='export LITELLM_API_KEY="sk-..."'
              copyKey="litellm-env"
            />
          </div>
        </FeatureCard>

        <FeatureCard
          icon={<ServerIcon className="text-emerald-600 dark:text-emerald-400" size={16} />}
          title="MCP Server Information"
          description="Connection details for your LiteLLM MCP server"
        >
          <CodeBlock title="Server URL" code={`${proxyBaseUrl}/mcp`} copyKey="litellm-server-url" />
        </FeatureCard>

        <FeatureCard
          icon={<Code className="text-emerald-600 dark:text-emerald-400" size={16} />}
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
            "server_url": "litellm_proxy",
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
      </div>
    </div>
  );

  const OpenAITab = () => (
    <div className="flex flex-col gap-6 w-full">
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-950/30 dark:to-indigo-950/30 p-6 rounded-lg border border-blue-100 dark:border-blue-900">
        <div className="flex items-center gap-3 mb-3">
          <Code className="text-primary" size={24} />
          <h4 className="mb-0 text-lg font-semibold text-blue-900 dark:text-blue-100">
            OpenAI Responses API Integration
          </h4>
        </div>
        <p className="text-blue-700 dark:text-blue-200">
          Connect OpenAI Responses API to your LiteLLM MCP server for seamless tool integration
        </p>
      </div>

      <div className="flex flex-col gap-6 w-full">
        <FeatureCard
          icon={<KeyIcon className="text-primary" size={16} />}
          title="API Key Setup"
          description="Configure your OpenAI API key for authentication"
        >
          <div className="flex flex-col gap-4 w-full">
            <div>
              <span className="text-foreground">
                Get your API key from the{" "}
                <a
                  href="https://platform.openai.com/api-keys"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary hover:text-primary/80 inline-flex items-center gap-1"
                >
                  OpenAI platform <ExternalLinkIcon size={12} />
                </a>
              </span>
            </div>
            <CodeBlock
              title="Environment Variable"
              code='export OPENAI_API_KEY="sk-..."'
              copyKey="openai-env"
            />
          </div>
        </FeatureCard>

        <FeatureCard
          icon={<ServerIcon className="text-primary" size={16} />}
          title="MCP Server Information"
          description="Connection details for your LiteLLM MCP server"
        >
          <CodeBlock title="Server URL" code={`${proxyBaseUrl}/mcp`} copyKey="openai-server-url" />
        </FeatureCard>

        <FeatureCard
          icon={<Code className="text-primary" size={16} />}
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
      </div>
    </div>
  );

  const CursorTab = () => (
    <div className="flex flex-col gap-6 w-full">
      <div className="bg-gradient-to-r from-purple-50 to-blue-50 dark:from-purple-950/30 dark:to-blue-950/30 p-6 rounded-lg border border-purple-100 dark:border-purple-900">
        <div className="flex items-center gap-3 mb-3">
          <Terminal className="text-purple-600 dark:text-purple-400" size={24} />
          <h4 className="mb-0 text-lg font-semibold text-purple-900 dark:text-purple-100">
            Cursor IDE Integration
          </h4>
        </div>
        <p className="text-purple-700 dark:text-purple-200">
          Use tools directly from Cursor IDE with LiteLLM MCP. Enable your AI assistant to perform
          real-world tasks without leaving your coding environment.
        </p>
      </div>

      <Card className="border border-border p-6">
        <h5 className="mb-4 text-foreground text-base font-semibold">Setup Instructions</h5>
        <div className="flex flex-col gap-6 w-full">
          <StepCard step={1} title="Open Cursor Settings">
            <span className="text-muted-foreground">
              Use the keyboard shortcut <code className="bg-muted px-2 py-1 rounded">⇧+⌘+J</code>{" "}
              (Mac) or <code className="bg-muted px-2 py-1 rounded">Ctrl+Shift+J</code>{" "}
              (Windows/Linux)
            </span>
          </StepCard>

          <StepCard step={2} title="Navigate to MCP Tools">
            <span className="text-muted-foreground">
              Go to the "MCP Tools" tab and click "New MCP Server"
            </span>
          </StepCard>

          <StepCard step={3} title="Add Configuration">
            <span className="text-muted-foreground mb-3 block">
              Copy the JSON configuration below and paste it into Cursor, then save with{" "}
              <code className="bg-muted px-2 py-1 rounded">Cmd+S</code> or{" "}
              <code className="bg-muted px-2 py-1 rounded">Ctrl+S</code>
            </span>
            <FeatureCard
              icon={<Code className="text-purple-600 dark:text-purple-400" size={16} />}
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
        </div>
      </Card>
    </div>
  );

  const StreamableHTTPTab = () => (
    <div className="flex flex-col gap-6 w-full">
      <div className="bg-gradient-to-r from-green-50 to-teal-50 dark:from-green-950/30 dark:to-teal-950/30 p-6 rounded-lg border border-green-100 dark:border-green-900">
        <div className="flex items-center gap-3 mb-3">
          <Globe className="text-green-600 dark:text-green-400" size={24} />
          <h4 className="mb-0 text-lg font-semibold text-green-900 dark:text-green-100">
            Streamable HTTP Transport
          </h4>
        </div>
        <p className="text-green-700 dark:text-green-200">
          Connect to LiteLLM MCP using HTTP transport. Compatible with any MCP client that supports HTTP
          streaming.
        </p>
      </div>

      <FeatureCard
        icon={<Globe className="text-green-600 dark:text-green-400" size={16} />}
        title="Universal MCP Connection"
        description="Use this URL with any MCP client that supports HTTP transport"
      >
        <div className="flex flex-col gap-4 w-full">
          <div>
            <span className="text-foreground">
              Each MCP client supports different transports. Refer to your client documentation to
              determine the appropriate transport method.
            </span>
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
            <a
              href="https://modelcontextprotocol.io/docs/concepts/transports"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 text-primary hover:text-primary/80"
            >
              Learn more about MCP transports
              <ExternalLinkIcon size={14} />
            </a>
          </div>
        </div>
      </FeatureCard>
    </div>
  );

  return (
    <div className="flex flex-col gap-6 w-full">
      <div>
        <h1 className="text-3xl font-bold text-foreground mb-3">Connect to your MCP client</h1>
        <p className="text-lg text-muted-foreground">
          Use tools directly from any MCP client with LiteLLM MCP. Enable your AI assistant to perform
          real-world tasks through a simple, secure connection.
        </p>
      </div>

      <Tabs defaultValue="openai" className="w-full">
        <TabsList className="mt-8 mb-6">
          <TabsTrigger value="openai" className="gap-2">
            <Code size={18} />
            OpenAI API
          </TabsTrigger>
          <TabsTrigger value="litellm" className="gap-2">
            <Zap size={18} />
            LiteLLM Proxy
          </TabsTrigger>
          <TabsTrigger value="cursor" className="gap-2">
            <Terminal size={18} />
            Cursor
          </TabsTrigger>
          <TabsTrigger value="http" className="gap-2">
            <Globe size={18} />
            Streamable HTTP
          </TabsTrigger>
        </TabsList>
        <TabsContent value="openai" className="mt-6">
          <OpenAITab />
        </TabsContent>
        <TabsContent value="litellm" className="mt-6">
          <LiteLLMProxyTab />
        </TabsContent>
        <TabsContent value="cursor" className="mt-6">
          <CursorTab />
        </TabsContent>
        <TabsContent value="http" className="mt-6">
          <StreamableHTTPTab />
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default MCPConnect;
