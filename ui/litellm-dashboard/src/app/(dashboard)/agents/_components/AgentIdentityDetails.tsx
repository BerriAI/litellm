import React from "react";
import type { components } from "@/lib/http/schema";
import { useQuery } from "@tanstack/react-query";
import { apiClient } from "@/components/networking";
import { Button } from "@/components/ui/button";
import { readAgentIdentity } from "./agent_identity";

const authenticationMessage = (error: boolean, lastAuthenticated?: string | null): string => {
  if (error) return "Could not load authentication evidence";
  if (lastAuthenticated) return `Last authenticated identity match: ${new Date(lastAuthenticated).toLocaleString()}`;
  return "Configured, awaiting an authenticated request";
};

const directoryStatus = (active?: boolean | null): string => {
  if (active == null) return "Unavailable";
  return active ? "Active" : "Inactive";
};

export const AgentIdentityDetails = ({
  agentId,
  identity: value,
  accessToken,
  isAdmin,
}: {
  agentId: string;
  identity: unknown;
  accessToken: string | null;
  isAdmin: boolean;
}) => {
  const identity = readAgentIdentity(value);
  const { data, isError, isFetching, refetch } = useQuery({
    queryKey: ["agent-identity", agentId, identity],
    queryFn: () =>
      apiClient.get<components["schemas"]["ManagedAgentIdentityStatus"]>(
        `/v1/agents/${encodeURIComponent(agentId)}/identity`,
        {
          accessToken: accessToken ?? "",
        },
      ),
    enabled: Boolean(isAdmin && accessToken && identity),
  });

  if (!identity || !isAdmin) return null;
  const executionLabel = data?.enabled ? "Enabled" : "Disabled";
  return (
    <section aria-label="Agent Identity" className="mb-6 space-y-2 rounded-lg border border-border p-4">
      <h3 className="font-medium">Agent Identity: Microsoft Entra ID</h3>
      <p className="text-sm">
        Tenant: <span className="font-mono">{identity.tenant_id}</span>
      </p>
      {identity.provisioning_source_id ? (
        <>
          <p className="text-sm">
            Entra Parent Identity ID: <span className="font-mono">{identity.client_id}</span>
          </p>
          <p className="text-sm">Provisioned through Entra SCIM</p>
          <p className="text-sm">Directory status: {directoryStatus(data?.directory_active)}</p>
          <p className="text-sm">Mapped access groups: {data?.directory_access_group_ids?.length ?? 0}</p>
        </>
      ) : (
        <>
          <p className="text-sm">
            Application (Client) ID: <span className="font-mono">{identity.client_id}</span>
          </p>
          <p className="text-sm">
            Enterprise application Object ID: {identity.service_principal_id || "Not configured"}
          </p>
        </>
      )}
      <p className="text-sm">
        Execution: {data ? executionLabel : "Loading"} · Mode: {data?.execution_mode ?? "Loading"}
      </p>
      <p className="text-sm">
        {data?.identity?.active === false
          ? "Identity unbound; execution is disabled"
          : authenticationMessage(isError, data?.last_authenticated_at)}
      </p>
      <p className="text-xs text-muted-foreground">
        Recent evidence comes from a validated Entra token matching this binding. It is persisted across restarts and
        cleared when the binding changes. Tool and model permissions are checked separately.
      </p>
      <div className="flex items-center gap-4">
        <Button
          variant="outline"
          size="sm"
          disabled={isFetching}
          onClick={() => {
            void refetch();
          }}
        >
          Refresh authentication evidence
        </Button>
        <a className="text-sm underline" href="/ui/logs/">
          View request logs
        </a>
      </div>
    </section>
  );
};
