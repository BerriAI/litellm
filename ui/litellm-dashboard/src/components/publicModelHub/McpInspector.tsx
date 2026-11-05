"use client";

import { useTranslation } from "react-i18next";

import type { MCPServerData } from "@/components/PublicModelHubTableColumns";

import { mcpAuthLabel } from "./hubFormat";
import { CodeBlock, EndpointField, Facts, PaneSection } from "./HubParts";
import { mcpEndpoint, mcpSnippet } from "./hubSnippets";
import { InspectorHeader } from "./InspectorHeader";

export function McpInspector({ server }: { server: MCPServerData }) {
  const { t } = useTranslation();
  return (
    <div className="flex flex-col gap-6 p-6">
      <InspectorHeader
        title={server.name}
        copyLabel={t("publicModelHub.mcpInspector.copyName")}
        description={server.mcp_info?.description}
      />
      <Facts
        groups={[
          [
            { label: t("publicModelHub.mcpInspector.facts.transport"), value: server.transport },
            { label: t("publicModelHub.mcpInspector.facts.signIn"), value: mcpAuthLabel(server.auth_type) },
            { label: t("publicModelHub.mcpInspector.facts.alias"), value: server.alias },
          ],
        ]}
      />
      <EndpointField id={`mcp-endpoint-${server.server_id}`} value={mcpEndpoint(server)} />
      <PaneSection title={t("publicModelHub.mcpInspector.connect")}>
        <CodeBlock code={mcpSnippet(server)} />
      </PaneSection>
    </div>
  );
}
