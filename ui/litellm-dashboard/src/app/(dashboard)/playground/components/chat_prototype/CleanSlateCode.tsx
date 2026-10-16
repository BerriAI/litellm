"use client";

import { useState } from "react";
import { Copy } from "lucide-react";
import { Button } from "@/components/ui/button";
import { FieldDescription } from "@/components/ui/field";
import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

const CODE_SAMPLES: Record<string, string> = {
  python: `from openai import OpenAI

client = OpenAI(
    api_key="sk-...",
    base_url="http://localhost:4000",
)

response = client.chat.completions.create(
    model="anthropic-sonnet-5",
    messages=[{
        "role": "user",
        "content": "What does this spec ask for?",
    }],
    tools=[
        {
            "type": "mcp",
            "server_label": "litellm",
            "server_url": "litellm_proxy/mcp/github",
            "require_approval": "never",
        },
        {
            "type": "mcp",
            "server_label": "litellm",
            "server_url": "litellm_proxy/mcp/linear",
            "require_approval": "never",
        },
    ],
    extra_body={
        "guardrails": ["pii-mask", "prompt-injection"],
        "vector_store_ids": ["support-docs"],
    },
)`,
  typescript: `import OpenAI from "openai";

const client = new OpenAI({
  apiKey: "sk-...",
  baseURL: "http://localhost:4000",
});

const response = await client.chat.completions.create({
  model: "anthropic-sonnet-5",
  messages: [{
    role: "user",
    content: "What does this spec ask for?",
  }],
  tools: [
    {
      type: "mcp",
      server_label: "litellm",
      server_url: "litellm_proxy/mcp/github",
      require_approval: "never",
    },
    {
      type: "mcp",
      server_label: "litellm",
      server_url: "litellm_proxy/mcp/linear",
      require_approval: "never",
    },
  ],
  guardrails: ["pii-mask", "prompt-injection"],
  vector_store_ids: ["support-docs"],
});`,
  curl: `curl http://localhost:4000/v1/chat/completions \\
  -H "Authorization: Bearer sk-..." \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "anthropic-sonnet-5",
    "messages": [{
      "role": "user",
      "content": "What does this spec ask for?"
    }],
    "tools": [
      {
        "type": "mcp",
        "server_label": "litellm",
        "server_url": "litellm_proxy/mcp/github",
        "require_approval": "never"
      },
      {
        "type": "mcp",
        "server_label": "litellm",
        "server_url": "litellm_proxy/mcp/linear",
        "require_approval": "never"
      }
    ],
    "guardrails": ["pii-mask", "prompt-injection"],
    "vector_store_ids": ["support-docs"]
  }'`,
};

export function CopyIconButton({ value, label }: { value: string; label: string }) {
  return (
    <Tooltip>
      <TooltipTrigger
        render={
          <Button
            variant="ghost"
            size="icon-xs"
            aria-label={label}
            onClick={() => void navigator.clipboard.writeText(value)}
          />
        }
      >
        <Copy />
      </TooltipTrigger>
      <TooltipContent>{label}</TooltipContent>
    </Tooltip>
  );
}

export function CodeTab() {
  const [language, setLanguage] = useState<string[]>(["python"]);
  const current = language[0] ?? "python";
  return (
    <div className="flex flex-col">
      <div className="flex items-center gap-2 px-4 pt-3">
        <ToggleGroup
          value={language}
          onValueChange={(value) => value.length && setLanguage(value as string[])}
          variant="outline"
          size="sm"
          spacing={0}
        >
          <ToggleGroupItem value="python" className="h-7 px-2 text-xs">
            Python
          </ToggleGroupItem>
          <ToggleGroupItem value="typescript" className="h-7 px-2 text-xs">
            TypeScript
          </ToggleGroupItem>
          <ToggleGroupItem value="curl" className="h-7 px-2 text-xs">
            cURL
          </ToggleGroupItem>
        </ToggleGroup>
        <span className="ml-auto">
          <CopyIconButton value={CODE_SAMPLES[current]} label="Copy code" />
        </span>
      </div>
      <FieldDescription className="px-4 pt-2 text-xs">
        Send this request from your own app. Replace sk-... with a virtual key.
      </FieldDescription>
      <ScrollArea className="w-full">
        <pre className="px-4 py-3 font-mono text-xs leading-5 whitespace-pre text-foreground">
          {CODE_SAMPLES[current]}
        </pre>
        <ScrollBar orientation="horizontal" />
      </ScrollArea>
    </div>
  );
}
