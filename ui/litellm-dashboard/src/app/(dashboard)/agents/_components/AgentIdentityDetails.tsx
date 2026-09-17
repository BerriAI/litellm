import React from "react";
import { useQuery } from "@tanstack/react-query";
import { apiClient } from "@/components/networking";
import { Button } from "@/components/ui/button";
import { readAgentIdentity } from "./agent_identity";

const authenticationMessage = (error: boolean, lastAuthenticated?: string | null): string => {
  if (error) return "Could not load authentication evidence";
  if (lastAuthenticated) return `Last authenticated identity match: ${new Date(lastAuthenticated).toLocaleString()}`;
  return "Configured, awaiting an authenticated request";
};

export const AgentIdentityDetails = ({
  agentId,
  identity: value,
  accessToken,
}: {
  agentId: string;
  identity: unknown;
  accessToken: string | null;
}) => {
  const identity = readAgentIdentity(value);
  const { data, isError, isFetching, refetch } = useQuery({
    queryKey: ["agent-identity", agentId, identity],
    queryFn: () =>
      apiClient.get<{ last_authenticated_at: string | null }>(`/v1/agents/${encodeURIComponent(agentId)}/identity`, {
        accessToken: accessToken ?? "",
      }),
    enabled: Boolean(accessToken && identity),
  });

  if (!identity) return null;
  return (
    <section aria-label="Agent Identity" className="mb-6 space-y-2 rounded-lg border border-border p-4">
      <h3 className="font-medium">Agent Identity: Microsoft Entra ID</h3>
      <p className="text-sm">
        Tenant: <span className="font-mono">{identity.tenant_id}</span>
      </p>
      <p className="text-sm">
        Application (Client) ID: <span className="font-mono">{identity.client_id}</span>
      </p>
      <p className="text-sm">{authenticationMessage(isError, data?.last_authenticated_at)}</p>
      <p className="text-xs text-muted-foreground">
        Recent evidence comes from a validated Entra token matching this binding. It is retained in the gateway cache
        for 24 hours and may clear on restart. Tool and model permissions are checked separately.
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
        <a className="text-sm underline" href="/ui/?page=logs">
          View request logs
        </a>
      </div>
    </section>
  );
};
