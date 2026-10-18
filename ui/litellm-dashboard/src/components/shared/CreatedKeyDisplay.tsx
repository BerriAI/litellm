import React, { useState } from "react";
import { CopyToClipboard } from "react-copy-to-clipboard";
import { ChevronDown, ChevronUp } from "lucide-react";
import anthropicLogo from "../../../public/assets/logos/anthropic.svg";
import openaiLogo from "../../../public/assets/logos/openai_small.svg";
import { getProxyBaseUrl } from "@/components/networking";
import { Button } from "@/components/ui/button";
import { toast } from "@/lib/toast";

interface CreatedKeyDisplayProps {
  apiKey: string;
}

export const buildAgentInstructions = (
  agent: "claude-code" | "codex",
  baseUrl: string,
  apiKey: string,
): string => {
  const normalizedBaseUrl = baseUrl.replace(/\/+$/, "");
  const agentLine =
    agent === "claude-code"
      ? `To run Claude Code through this gateway, set ANTHROPIC_BASE_URL=${normalizedBaseUrl} and ` +
        `ANTHROPIC_AUTH_TOKEN=${apiKey}.`
      : `To run Codex through this gateway, set OPENAI_BASE_URL=${normalizedBaseUrl}/v1 and ` +
        `OPENAI_API_KEY=${apiKey}.`;
  const endpointIntroduction =
    "LiteLLM is a gateway that exposes OpenAI and Anthropic compatible APIs. " +
    `Send the virtual key as a Bearer token (Authorization: Bearer ${apiKey}) to:`;

  return [
    "Use this LiteLLM AI gateway for model inference.",
    "",
    `Gateway URL: ${normalizedBaseUrl}`,
    `Virtual key: ${apiKey}`,
    "",
    endpointIntroduction,
    `- POST ${normalizedBaseUrl}/v1/chat/completions (OpenAI Chat Completions)`,
    `- POST ${normalizedBaseUrl}/v1/responses (OpenAI Responses)`,
    `- POST ${normalizedBaseUrl}/v1/messages (Anthropic Messages)`,
    `- GET ${normalizedBaseUrl}/v1/models (models this key can use)`,
    "",
    agentLine,
    "",
    "Docs for everything else LiteLLM supports: https://docs.litellm.ai/docs/",
  ].join("\n");
};

/**
 * Shared component for displaying a newly-created virtual key.
 * Used on the Virtual Keys page and in the Add Agent wizard.
 */
const CreatedKeyDisplay: React.FC<CreatedKeyDisplayProps> = ({ apiKey }) => {
  const [copied, setCopied] = useState(false);
  const [instructionsCopied, setInstructionsCopied] = useState(false);
  const [selectedAgent, setSelectedAgent] = useState<"claude-code" | "codex">("claude-code");
  const [showInstructions, setShowInstructions] = useState(false);
  const baseUrl = getProxyBaseUrl().replace(/\/+$/, "");
  const instructions = buildAgentInstructions(selectedAgent, baseUrl, apiKey);
  const previewText = instructions.split("\n").slice(0, 3).join("\n");
  const instructionsTitle =
    selectedAgent === "claude-code" ? "Instructions for Claude Code" : "Instructions for Codex";

  const handleCopy = () => {
    setCopied(true);
    toast.success("Key copied to clipboard");
    setTimeout(() => setCopied(false), 2000);
  };

  const handleCopyInstructions = () => {
    setInstructionsCopied(true);
    toast.success("Instructions copied");
    setTimeout(() => setInstructionsCopied(false), 2000);
  };

  return (
    <div>
      <p className="mb-2">
        Please save this secret key somewhere safe and accessible. For security reasons,{" "}
        <b>you will not be able to view it again</b> through your LiteLLM account. If you lose this secret key, you will
        need to generate a new one.
      </p>

      <p className="text-sm text-muted-foreground mt-3 mb-1">Virtual Key:</p>
      <div className="bg-muted rounded-md p-2.5 mb-2.5">
        <pre className="m-0 whitespace-normal break-words text-foreground">{apiKey}</pre>
      </div>

      <CopyToClipboard text={apiKey} onCopy={handleCopy}>
        <Button className="mt-3">{copied ? "Copied!" : "Copy Virtual Key"}</Button>
      </CopyToClipboard>

      <p className="text-sm text-muted-foreground mt-4 mb-1">Gateway URL:</p>
      <div className="bg-muted rounded-md p-2.5 mb-2.5 flex items-center justify-between gap-2">
        <pre className="m-0 whitespace-normal break-words text-foreground">{baseUrl}</pre>
        <CopyToClipboard text={baseUrl} onCopy={() => toast.success("Gateway URL copied to clipboard")}>
          <Button type="button" variant="outline" className="shrink-0">
            Copy URL
          </Button>
        </CopyToClipboard>
      </div>

      <p className="text-sm text-muted-foreground mt-4 mb-1">Set up your agent</p>
      <div className="flex flex-wrap gap-2">
        <Button
          type="button"
          variant={selectedAgent === "claude-code" ? "default" : "outline"}
          aria-pressed={selectedAgent === "claude-code"}
          onClick={() => setSelectedAgent("claude-code")}
        >
          <img src={anthropicLogo.src} alt="" className="size-4 object-contain" />
          Claude Code
        </Button>
        <Button
          type="button"
          variant={selectedAgent === "codex" ? "default" : "outline"}
          aria-pressed={selectedAgent === "codex"}
          onClick={() => setSelectedAgent("codex")}
        >
          <img src={openaiLogo.src} alt="" className="size-4 object-contain" />
          Codex
        </Button>
      </div>

      <div className="mt-3 rounded-md border border-border p-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <p className="text-sm font-medium">{instructionsTitle}</p>
          <div className="flex items-center gap-2">
            <CopyToClipboard text={instructions} onCopy={handleCopyInstructions}>
              <Button type="button">{instructionsCopied ? "Copied!" : "Copy instructions"}</Button>
            </CopyToClipboard>
            <Button
              type="button"
              variant="outline"
              aria-expanded={showInstructions}
              onClick={() => setShowInstructions((expanded) => !expanded)}
            >
              {showInstructions ? "Hide" : "Show"}
              {showInstructions ? (
                <ChevronUp aria-hidden="true" className="size-4" />
              ) : (
                <ChevronDown aria-hidden="true" className="size-4" />
              )}
            </Button>
          </div>
        </div>
        <pre className="mt-3 whitespace-pre-wrap break-words text-sm text-foreground">
          {showInstructions ? instructions : <span className="text-muted-foreground opacity-70">{previewText}</span>}
        </pre>
      </div>
    </div>
  );
};

export default CreatedKeyDisplay;
