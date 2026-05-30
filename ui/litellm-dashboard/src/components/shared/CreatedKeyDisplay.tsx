import React, { useMemo, useState } from "react";
import { CopyToClipboard } from "react-copy-to-clipboard";
import { Button, Tooltip } from "antd";
import { CopyOutlined, WarningOutlined } from "@ant-design/icons";
import MessageManager from "@/components/molecules/message_manager";
import { proxyBaseUrl } from "@/components/networking";

interface CreatedKeyDisplayProps {
  apiKey: string;
}

const codingAgentLogos: { name: string; src: string }[] = [
  { name: "Cursor", src: "../ui/assets/logos/cursor.svg" },
  { name: "Claude Code", src: "../ui/assets/logos/anthropic.svg" },
  { name: "OpenAI Codex", src: "../ui/assets/logos/openai_small.svg" },
  { name: "GitHub Copilot", src: "../ui/assets/logos/github_copilot.svg" },
];

const CodingAgentLogos: React.FC = () => (
  <div
    className="flex items-center"
    aria-label="Compatible coding agents"
    data-testid="coding-agent-logos"
  >
    {codingAgentLogos.map((logo, index) => (
      <img
        key={logo.name}
        src={logo.src}
        alt={logo.name}
        title={logo.name}
        style={{
          width: 18,
          height: 18,
          borderRadius: "50%",
          background: "#fff",
          border: "1px solid #e5e7eb",
          padding: 2,
          marginLeft: index === 0 ? 0 : -6,
          objectFit: "contain",
          boxSizing: "border-box",
        }}
      />
    ))}
  </div>
);

const buildCodingAgentPrompt = (apiKey: string, baseUrl: string): string => {
  return `You have access to LiteLLM, an OpenAI-compatible AI gateway that lets you call 100+ LLMs (OpenAI, Anthropic, Gemini, Bedrock, etc.) through a single API.

Use these credentials:
- Base URL: ${baseUrl}
- API key: ${apiKey}

LiteLLM is a drop-in replacement for the OpenAI SDK. Point the SDK's \`base_url\` at the URL above and use the key as your \`OPENAI_API_KEY\`. To list the models available to this key, GET ${baseUrl}/v1/models with header \`Authorization: Bearer ${apiKey}\`.

LiteLLM docs:
- llms.txt (overview + all doc links): https://docs.litellm.ai/llms.txt
- llms-full.txt (complete reference with inline code examples): https://docs.litellm.ai/llms-full.txt`;
};

const resolveProxyBaseUrl = (): string => {
  if (proxyBaseUrl) return proxyBaseUrl.replace(/\/$/, "");
  if (typeof window !== "undefined" && window.location?.origin) {
    return window.location.origin;
  }
  return "https://your-litellm-proxy";
};

const CreatedKeyDisplay: React.FC<CreatedKeyDisplayProps> = ({ apiKey }) => {
  const [copiedKey, setCopiedKey] = useState(false);
  const [copiedPrompt, setCopiedPrompt] = useState(false);

  const codingAgentPrompt = useMemo(
    () => buildCodingAgentPrompt(apiKey, resolveProxyBaseUrl()),
    [apiKey],
  );

  const handleCopyKey = () => {
    setCopiedKey(true);
    MessageManager.success("Key copied to clipboard");
    setTimeout(() => setCopiedKey(false), 2000);
  };

  const handleCopyPrompt = () => {
    setCopiedPrompt(true);
    MessageManager.success("Prompt copied to clipboard");
    setTimeout(() => setCopiedPrompt(false), 2000);
  };

  return (
    <div className="created-key-display">
      <div className="mb-2">
        <h2 className="text-lg font-semibold m-0">API Key Created</h2>
        <p className="text-sm text-gray-500 mt-1 mb-0">
          Paste this prompt into any coding agent to start using LiteLLM.
        </p>
      </div>

      <div
        className="flex items-start gap-2 px-3 py-2 mb-4 rounded-md border"
        style={{ background: "#fffbeb", borderColor: "#fde68a" }}
        role="alert"
      >
        <WarningOutlined style={{ color: "#b45309", marginTop: 3 }} />
        <span className="text-sm" style={{ color: "#92400e" }}>
          Make sure to copy your API key now. You won&apos;t be able to see it again.
        </span>
      </div>

      <div
        className="rounded-md border mb-4"
        style={{ borderColor: "#e5e7eb", background: "#fafafa" }}
      >
        <div className="flex items-center justify-between px-3 pt-3 pb-1">
          <span className="text-sm font-medium text-gray-700">Your API Key</span>
          <Tooltip title={copiedKey ? "Copied!" : "Copy API key"}>
            <CopyToClipboard text={apiKey} onCopy={handleCopyKey}>
              <Button
                type="text"
                size="small"
                aria-label="Copy API key"
                icon={<CopyOutlined />}
              />
            </CopyToClipboard>
          </Tooltip>
        </div>
        <div className="px-3 pb-3">
          <pre
            className="m-0 text-sm"
            style={{
              wordBreak: "break-all",
              whiteSpace: "pre-wrap",
              fontFamily:
                "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace",
            }}
            data-testid="created-key-value"
          >
            {apiKey}
          </pre>
        </div>
      </div>

      <div
        className="rounded-md border"
        style={{ borderColor: "#e5e7eb", background: "#fafafa" }}
      >
        <div className="flex items-center justify-between px-3 pt-3 pb-1">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-gray-700">
              Prompt for coding agents
            </span>
            <CodingAgentLogos />
          </div>
          <Tooltip title={copiedPrompt ? "Copied!" : "Copy prompt"}>
            <CopyToClipboard text={codingAgentPrompt} onCopy={handleCopyPrompt}>
              <Button
                type="text"
                size="small"
                aria-label="Copy prompt for coding agents"
                icon={<CopyOutlined />}
              />
            </CopyToClipboard>
          </Tooltip>
        </div>
        <div className="px-3 pb-3">
          <pre
            className="m-0 text-sm text-gray-700"
            style={{
              maxHeight: 200,
              overflowY: "auto",
              whiteSpace: "pre-wrap",
              wordBreak: "break-word",
              fontFamily:
                "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace",
            }}
            data-testid="coding-agent-prompt"
          >
            {codingAgentPrompt}
          </pre>
        </div>
      </div>

      <CopyToClipboard text={apiKey} onCopy={handleCopyKey}>
        <Button type="primary" style={{ marginTop: 16 }}>
          {copiedKey ? "Copied!" : "Copy Virtual Key"}
        </Button>
      </CopyToClipboard>
    </div>
  );
};

export default CreatedKeyDisplay;
