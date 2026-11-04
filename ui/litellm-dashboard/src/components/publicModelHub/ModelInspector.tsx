"use client";

import { ChevronDown } from "lucide-react";
import * as React from "react";

import type { ModelGroupInfo } from "@/components/PublicModelHubTableColumns";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

import {
  formatPerMillion,
  formatTokens,
  modeLabel,
  modelCapabilities,
  providerName,
  wildcardExample,
} from "./hubFormat";
import { Badges, CodeBlock, type Fact, Facts, PaneSection, ProviderAvatar, ProviderName } from "./HubParts";
import { curlSnippet, modelSnippet } from "./hubSnippets";
import { HealthBadge } from "./HubStatus";
import { InspectorHeader } from "./InspectorHeader";

const summary = (model: ModelGroupInfo): string | undefined => {
  if (model.description) return model.description;
  const providers = (model.providers ?? []).map(providerName).join(", ");
  return model.mode && providers ? `${modeLabel(model.mode)} model from ${providers}` : undefined;
};

const tokens = (value: number | undefined) => (value ? `${formatTokens(value)} tokens` : null);

const modelFacts = (model: ModelGroupInfo): Fact[][] => {
  const providers = model.providers ?? [];
  const tags = model.custom_tags ?? [];
  return [
    [
      { label: "Mode", value: model.mode ? modeLabel(model.mode) : null },
      { label: "Context", value: tokens(model.max_input_tokens) },
      { label: "Max output", value: tokens(model.max_output_tokens) },
    ],
    [
      { label: "Input / 1M tokens", value: formatPerMillion(model.input_cost_per_token) },
      { label: "Output / 1M tokens", value: formatPerMillion(model.output_cost_per_token) },
    ],
    [
      { label: "Requests per minute", value: model.rpm },
      { label: "Tokens per minute", value: model.tpm ? formatTokens(model.tpm) : null },
    ],
    [
      {
        label: providers.length > 1 ? "Providers" : "Provider",
        wide: true,
        value: providers.length > 0 && (
          <span className="flex flex-wrap gap-x-4 gap-y-1">
            {providers.map((provider) => (
              <ProviderName key={provider} provider={provider} />
            ))}
          </span>
        ),
      },
      { label: "Tags", wide: true, value: tags.length > 0 && <Badges items={tags} /> },
    ],
  ];
};

function CallThisModel({ model }: { model: ModelGroupInfo }) {
  const curl = curlSnippet(model);
  if (!curl) {
    return (
      <PaneSection title="Call this model">
        <CodeBlock code={modelSnippet(model)} />
      </PaneSection>
    );
  }
  return (
    <PaneSection title="Call this model">
      <Tabs defaultValue="python" className="gap-3">
        <TabsList>
          <TabsTrigger value="python">Python</TabsTrigger>
          <TabsTrigger value="curl">cURL</TabsTrigger>
        </TabsList>
        <TabsContent value="python">
          <CodeBlock code={modelSnippet(model)} />
        </TabsContent>
        <TabsContent value="curl">
          <CodeBlock code={curl} />
        </TabsContent>
      </Tabs>
    </PaneSection>
  );
}

function CustomMetadata({ model }: { model: ModelGroupInfo }) {
  const metadata = Object.entries(model.custom_metadata ?? {});
  if (metadata.length === 0) return null;
  return (
    <PaneSection title="Custom metadata">
      <dl className="grid grid-cols-[160px_minmax(0,1fr)] gap-x-4 gap-y-2 text-sm">
        {metadata.map(([key, value]) => (
          <React.Fragment key={key}>
            <dt className="text-xs font-medium text-muted-foreground">{key}</dt>
            <dd>{String(value)}</dd>
          </React.Fragment>
        ))}
      </dl>
    </PaneSection>
  );
}

function SupportedParams({ params }: { params: string[] }) {
  if (params.length === 0) return null;
  return (
    <Collapsible>
      <PaneSection
        title="Supported OpenAI parameters"
        aside={
          <CollapsibleTrigger render={<Button variant="ghost" size="sm" />}>
            Show {params.length}
            <ChevronDown className="text-muted-foreground" />
          </CollapsibleTrigger>
        }
      >
        <CollapsibleContent>
          <Badges items={params} mono />
        </CollapsibleContent>
      </PaneSection>
    </Collapsible>
  );
}

export function ModelInspector({ model }: { model: ModelGroupInfo }) {
  const providers = model.providers ?? [];
  const capabilities = modelCapabilities(model);
  return (
    <div className="flex flex-col gap-6 p-6">
      <InspectorHeader
        media={<ProviderAvatar provider={providers[0] ?? ""} className="size-6" />}
        title={model.model_group}
        copyLabel="Copy model name"
        status={<HealthBadge model={model} />}
        description={summary(model)}
      />
      {model.model_group.includes("*") && (
        <Alert>
          <AlertTitle>Wildcard model</AlertTitle>
          <AlertDescription>
            Call it as <code className="font-mono">{wildcardExample(model.model_group)}</code>, with any model name this
            provider offers
          </AlertDescription>
        </Alert>
      )}
      <Facts groups={modelFacts(model)} />
      {capabilities.length > 0 && (
        <PaneSection title="Capabilities">
          <Badges items={capabilities} />
        </PaneSection>
      )}
      <CallThisModel model={model} />
      <CustomMetadata model={model} />
      <SupportedParams params={model.supported_openai_params ?? []} />
    </div>
  );
}
