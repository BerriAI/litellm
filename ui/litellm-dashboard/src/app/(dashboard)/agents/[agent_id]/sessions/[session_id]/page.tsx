"use client";

/**
 * /agents/{agent_id}/sessions/{session_id} — three-pane session view.
 *
 * Reflects the LIT-2881 layout:
 *   - left: SessionList scoped to the active Agent
 *   - middle: Conversation (snapshot + live SSE) + Composer
 *   - right: Git / Terminal tabs
 */
import { use } from "react";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import ThreePane from "@/app/(dashboard)/agents/components/three-pane";

interface PageProps {
  params: Promise<{ agent_id: string; session_id: string }>;
}

export default function SessionThreePanePage({ params }: PageProps) {
  const { agent_id, session_id } = use(params);
  const { accessToken } = useAuthorized();
  return <ThreePane agentId={agent_id} sessionId={session_id} accessToken={accessToken} />;
}
