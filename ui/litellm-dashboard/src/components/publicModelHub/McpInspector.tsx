"use client";

import type { MCPServerData } from "@/components/PublicModelHubTableColumns";

import { mcpAuthLabel } from "./hubFormat";
import { CodeBlock, EndpointField, Facts, PaneSection } from "./HubParts";
import { mcpEndpoint, mcpSnippet } from "./hubSnippets";
import { InspectorHeader } from "./InspectorHeader";

export function McpInspector({ server }: { server: MCPServerData }) {
  return (
    <div className="flex flex-col gap-6 p-6">
      <InspectorHeader
        title={server.server_name}
        copyLabel="Copy server name"
        description={server.mcp_info?.description}
      />
      <Facts
        groups={[
          [
            { label: "Transport", value: server.transport },
            { label: "Sign-in", value: mcpAuthLabel(server.auth_type) },
            { label: "Alias", value: server.alias },
          ],
        ]}
      />
      <EndpointField id={`mcp-endpoint-${server.server_id}`} value={mcpEndpoint(server)} />
      <PaneSection title="Connect a client">
        <CodeBlock code={mcpSnippet(server)} />
      </PaneSection>
    </div>
  );
}
