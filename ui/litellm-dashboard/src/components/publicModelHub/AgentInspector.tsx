"use client";

import { ExternalLink } from "lucide-react";

import type { AgentCard } from "@/components/PublicModelHubTableColumns";
import { Button } from "@/components/ui/button";
import { Field, FieldDescription } from "@/components/ui/field";
import { Item, ItemActions, ItemContent, ItemDescription, ItemGroup, ItemTitle } from "@/components/ui/item";

import { agentCapabilities } from "./hubFormat";
import { Badges, CodeBlock, EndpointField, Facts, PaneSection } from "./HubParts";
import { agentSnippets } from "./hubSnippets";
import { InspectorHeader } from "./InspectorHeader";

const modes = (agent: AgentCard) =>
  `${(agent.defaultInputModes ?? []).join(", ")} / ${(agent.defaultOutputModes ?? []).join(", ")}`;

export function AgentInspector({ agent }: { agent: AgentCard }) {
  const capabilities = agentCapabilities(agent);
  return (
    <div className="flex flex-col gap-6 p-6">
      <InspectorHeader title={agent.name} copyLabel="Copy agent name" description={agent.description} />
      <Facts
        groups={[
          [
            { label: "Version", value: agent.version },
            { label: "Owner", value: agent.provider?.organization },
            { label: "Protocol", value: agent.protocolVersion && `A2A ${agent.protocolVersion}` },
          ],
          [
            { label: "Input / output", wide: true, value: modes(agent) },
            { label: "Capabilities", wide: true, value: capabilities.length > 0 && <Badges items={capabilities} /> },
          ],
        ]}
      />
      <EndpointField id={`agent-endpoint-${agent.name}`} value={agent.url} />
      {agent.documentationUrl && (
        <Button
          variant="link"
          nativeButton={false}
          className="h-auto self-start px-0"
          render={<a href={agent.documentationUrl} target="_blank" rel="noopener noreferrer" />}
        >
          Read the documentation
          <ExternalLink />
        </Button>
      )}
      {(agent.skills ?? []).length > 0 && (
        <PaneSection title="Skills">
          <ItemGroup className="gap-2">
            {agent.skills.map((skill) => (
              <Item key={skill.id} variant="outline" size="sm" role="listitem">
                <ItemContent>
                  <ItemTitle>{skill.name}</ItemTitle>
                  <ItemDescription>{skill.description}</ItemDescription>
                </ItemContent>
                <ItemActions>
                  <Badges items={skill.tags ?? []} />
                </ItemActions>
              </Item>
            ))}
          </ItemGroup>
        </PaneSection>
      )}
      <PaneSection title="Call this agent">
        {agentSnippets(agent).map((snippet, index) => (
          <Field key={snippet.title}>
            <FieldDescription>
              Step {index + 1}: {snippet.title}
            </FieldDescription>
            <CodeBlock code={snippet.code} />
          </Field>
        ))}
      </PaneSection>
    </div>
  );
}
