"use client";

/**
 * /agents/{agent_id} — per-agent landing page.
 *
 * Shows the agent's identity, system prompt, and sessions list. + New
 * Session opens the create dialog and redirects into the three-pane view.
 */
import { use } from "react";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import AgentDetail from "@/app/(dashboard)/agents/components/agent-detail";

interface PageProps {
  params: Promise<{ agent_id: string }>;
}

export default function AgentDetailPage({ params }: PageProps) {
  const { agent_id } = use(params);
  const { accessToken } = useAuthorized();
  return <AgentDetail agentId={agent_id} accessToken={accessToken} />;
}
