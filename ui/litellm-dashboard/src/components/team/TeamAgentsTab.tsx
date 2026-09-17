import React, { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { apiClient } from "@/components/networking";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { readAgentIdentity } from "@/app/(dashboard)/agents/_components/agent_identity";

interface TeamAgent {
  agent_id: string;
  agent_name: string;
  litellm_params?: { team_id?: string | null; identity?: unknown } | null;
}

export default function TeamAgentsTab({
  teamId,
  accessToken,
  canManage,
}: {
  teamId: string;
  accessToken: string | null;
  canManage: boolean;
}) {
  const [selected, setSelected] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const {
    data: agents = [],
    isSuccess: loaded,
    isError: loadFailed,
    refetch,
  } = useQuery({
    queryKey: ["team-agent-memberships", teamId],
    queryFn: () => apiClient.get<TeamAgent[]>("/v1/agents", { accessToken: accessToken ?? "" }),
    enabled: Boolean(accessToken && canManage),
  });
  const displayedError =
    error || (loadFailed ? "Could not load agent memberships. Retry before making changes." : null);

  const assign = async (agentId: string, destination: string | null) => {
    if (!accessToken) return;
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      await apiClient.put(`/v1/agents/${encodeURIComponent(agentId)}/team`, {
        accessToken,
        body: { team_id: destination },
      });
      setSelected(null);
      setNotice(
        destination
          ? "Agent added to this team"
          : "Agent removed. JWT requests are denied until it is assigned to a team.",
      );
      await refetch();
    } catch {
      setError("Could not update agent membership. Refresh and try again.");
    } finally {
      setBusy(false);
    }
  };

  if (!canManage) return <p>A proxy administrator manages agent team assignments.</p>;
  const members = agents.filter((agent) => agent.litellm_params?.team_id === teamId);
  const available = agents.filter((agent) => !agent.litellm_params?.team_id);
  const matches = available.filter((agent) =>
    `${agent.agent_name} ${agent.agent_id}`.toLowerCase().includes(search.toLowerCase()),
  );
  const chosen = available.find((agent) => agent.agent_id === selected);
  const identity = readAgentIdentity(chosen?.litellm_params?.identity);

  return (
    <section aria-label="Team agents" className="space-y-4 rounded-lg border border-border p-6">
      <h3 className="text-lg font-medium">Agents in this team</h3>
      <p className="text-sm text-muted-foreground">
        Assigned agents use this team&apos;s controls for JWT requests. User and agent restrictions also apply. Each
        agent belongs to one team; remove it from its current team before assigning another.
      </p>
      {displayedError && (
        <p role="alert" className="text-sm text-destructive">
          {displayedError}
        </p>
      )}
      {notice && (
        <p role="status" className="text-sm">
          {notice}
        </p>
      )}
      <Button
        variant="outline"
        disabled={busy}
        onClick={() => {
          void refetch();
        }}
      >
        Refresh agents
      </Button>
      {!loaded && !displayedError && <p>Loading agents...</p>}
      {loaded && members.length === 0 && <p>No agents assigned to this team</p>}
      {members.map((agent) => {
        const binding = readAgentIdentity(agent.litellm_params?.identity);
        return (
          <article
            key={agent.agent_id}
            aria-label={agent.agent_name}
            className="space-y-2 rounded-md border border-border p-4"
          >
            <h4 className="font-medium">{agent.agent_name}</h4>
            <p className="text-sm">
              LiteLLM Agent ID: <code>{agent.agent_id}</code>
            </p>
            {binding && (
              <>
                <p className="text-sm">Identity provider: Microsoft Entra ID</p>
                <p className="text-sm">
                  Tenant: <code>{binding.tenant_id}</code>
                </p>
                <p className="text-sm">
                  Application (Client) ID: <code>{binding.client_id}</code>
                </p>
              </>
            )}
            <Button
              variant="outline"
              disabled={busy}
              onClick={() => {
                void assign(agent.agent_id, null);
              }}
            >
              Remove {agent.agent_name} from team
            </Button>
          </article>
        );
      })}
      <div className="space-y-3 border-t border-border pt-4">
        <h4 className="font-medium">Add an existing agent</h4>
        <Input
          aria-label="Search unassigned agents"
          placeholder="Search by agent name or LiteLLM Agent ID"
          value={search}
          onChange={(event) => setSearch(event.target.value)}
        />
        <Select value={selected} onValueChange={setSelected}>
          <SelectTrigger aria-label="Agent to add">
            <SelectValue placeholder="Select an unassigned agent" />
          </SelectTrigger>
          <SelectContent>
            {matches.map((agent) => (
              <SelectItem key={agent.agent_id} value={agent.agent_id}>
                {agent.agent_name} ({agent.agent_id})
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        {loaded && available.length === 0 && (
          <p className="text-sm text-muted-foreground">
            No unassigned agents. Register an agent in Agents, or remove an existing assignment first.
          </p>
        )}
        {identity && (
          <p className="text-sm">
            Microsoft Entra ID · Tenant: {identity.tenant_id} · Application (Client) ID: {identity.client_id}
          </p>
        )}
        <Button
          disabled={busy || !loaded || !selected}
          onClick={() => {
            if (selected) void assign(selected, teamId);
          }}
        >
          Add agent to team
        </Button>
      </div>
    </section>
  );
}
