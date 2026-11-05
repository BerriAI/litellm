"use client";

import { ExternalLink } from "lucide-react";
import { useTranslation } from "react-i18next";

import type { AgentCard } from "@/components/PublicModelHubTableColumns";
import { Button } from "@/components/ui/button";
import { Field, FieldDescription } from "@/components/ui/field";
import { Item, ItemActions, ItemContent, ItemDescription, ItemGroup, ItemTitle } from "@/components/ui/item";

import { agentCapabilities } from "./hubFormat";
import { Badges, CodeBlock, EndpointField, Facts, PaneSection } from "./HubParts";
import { agentSnippets } from "./hubSnippets";
import { InspectorHeader } from "./InspectorHeader";

const modes = (agent: AgentCard) => {
  const input = (agent.defaultInputModes ?? []).join(", ");
  const output = (agent.defaultOutputModes ?? []).join(", ");
  return input || output ? `${input || "-"} / ${output || "-"}` : null;
};

export function AgentInspector({ agent }: { agent: AgentCard }) {
  const { t } = useTranslation();
  const capabilities = agentCapabilities(agent);
  return (
    <div className="flex flex-col gap-6 p-6">
      <InspectorHeader
        title={agent.name}
        copyLabel={t("publicModelHub.agentInspector.copyName")}
        description={agent.description}
      />
      <Facts
        groups={[
          [
            { label: t("publicModelHub.agentInspector.facts.version"), value: agent.version },
            { label: t("publicModelHub.agentInspector.facts.owner"), value: agent.provider?.organization },
            {
              label: t("publicModelHub.agentInspector.facts.protocol"),
              value: agent.protocolVersion && `A2A ${agent.protocolVersion}`,
            },
          ],
          [
            { label: t("publicModelHub.agentInspector.facts.inputOutput"), wide: true, value: modes(agent) },
            {
              label: t("publicModelHub.agentInspector.facts.capabilities"),
              wide: true,
              value: capabilities.length > 0 && <Badges items={capabilities} />,
            },
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
          {t("publicModelHub.agentInspector.documentation")}
          <ExternalLink />
        </Button>
      )}
      {(agent.skills ?? []).length > 0 && (
        <PaneSection title={t("publicModelHub.agentInspector.skills")}>
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
      <PaneSection title={t("publicModelHub.agentInspector.callAgent")}>
        {agentSnippets(agent).map((snippet, index) => (
          <Field key={snippet.title}>
            <FieldDescription>
              {t("publicModelHub.agentInspector.step", { step: index + 1, title: snippet.title })}
            </FieldDescription>
            <CodeBlock code={snippet.code} />
          </Field>
        ))}
      </PaneSection>
    </div>
  );
}
