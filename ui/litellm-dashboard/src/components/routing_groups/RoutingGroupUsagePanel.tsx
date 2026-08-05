"use client";

import { Code2 } from "lucide-react";
import React from "react";

import CodeBlock from "@/components/CodeBlock";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

import { formatStrategyLabel } from "./strategy";
import type { RoutingGroup } from "./types";

interface RoutingGroupUsagePanelProps {
  group: RoutingGroup;
  baseUrl: string;
}

const exampleModel = (group: RoutingGroup): string => group.models[0] ?? "<your-model>";

const buildCurlSnippet = (group: RoutingGroup, baseUrl: string): string =>
  `curl -X POST '${baseUrl}/v1/chat/completions' \\
  -H 'Content-Type: application/json' \\
  -H 'Authorization: Bearer $LITELLM_API_KEY' \\
  -d '{
    "model": "${exampleModel(group)}",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'`;

const buildPythonSnippet = (group: RoutingGroup, baseUrl: string): string =>
  `from openai import OpenAI

client = OpenAI(
    api_key="$LITELLM_API_KEY",
    base_url="${baseUrl}",
)

response = client.chat.completions.create(
    model="${exampleModel(group)}",
    messages=[{"role": "user", "content": "Hello!"}],
)

print(response)`;

const buildJsSnippet = (group: RoutingGroup, baseUrl: string): string =>
  `import OpenAI from "openai";

const client = new OpenAI({
  apiKey: process.env.LITELLM_API_KEY,
  baseURL: "${baseUrl}",
});

const response = await client.chat.completions.create({
  model: "${exampleModel(group)}",
  messages: [{ role: "user", content: "Hello!" }],
});

console.log(response);`;

const SNIPPET_TABS = [
  { value: "curl", label: "cURL", language: "bash", build: buildCurlSnippet },
  { value: "python", label: "Python (OpenAI SDK)", language: "python", build: buildPythonSnippet },
  { value: "javascript", label: "JavaScript (OpenAI SDK)", language: "javascript", build: buildJsSnippet },
] as const;

export function RoutingGroupUsagePanel({ group, baseUrl }: RoutingGroupUsagePanelProps) {
  return (
    <div className="border-y bg-muted/40 px-4 py-4">
      <div className="mb-2 flex items-center gap-2">
        <Code2 className="size-4 text-primary" />
        <span className="text-sm font-medium text-foreground">How routing works for this group</span>
      </div>
      <p className="mb-3 text-sm text-muted-foreground">
        Callers request any model in the group by name; LiteLLM picks a deployment behind the scenes using the{" "}
        <span className="font-medium text-foreground">{formatStrategyLabel(group.routing_strategy)}</span> strategy.
      </p>
      <Tabs defaultValue="curl">
        <TabsList variant="line" className="h-auto w-full justify-start rounded-none border-b p-0">
          {SNIPPET_TABS.map((tab) => (
            <TabsTrigger key={tab.value} value={tab.value} className="flex-none rounded-none px-4 py-2">
              {tab.label}
            </TabsTrigger>
          ))}
        </TabsList>
        {SNIPPET_TABS.map((tab) => (
          <TabsContent key={tab.value} value={tab.value} className="pt-3">
            <CodeBlock language={tab.language} code={tab.build(group, baseUrl)} />
          </TabsContent>
        ))}
      </Tabs>
    </div>
  );
}
