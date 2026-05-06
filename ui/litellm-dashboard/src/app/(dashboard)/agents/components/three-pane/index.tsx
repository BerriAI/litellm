"use client";

/**
 * ThreePane — orchestrator for the session view.
 *
 * Lays out:
 *   - left:  SessionList (scoped to the active Agent)
 *   - middle: Conversation (initial snapshot + live SSE events)
 *   - right: RightPanel (Git / Terminal tabs)
 *
 * Owns the SSE subscription and the conversation snapshot fetch. Children
 * are presentational — they receive `events` and the snapshot as props.
 */
import { useEffect, useState } from "react";
import { Spin, message } from "antd";
import SessionList from "@/app/(dashboard)/agents/components/session-list";
import Conversation from "@/app/(dashboard)/agents/components/three-pane/conversation";
import RightPanel from "@/app/(dashboard)/agents/components/three-pane/right-panel";
import NewSessionDialog from "@/app/(dashboard)/agents/components/new-session-dialog";
import { useSessionEventStream } from "@/app/(dashboard)/agents/hooks/useSessionEventStream";
import { getCloudRun, getSessionConversation, listCloudSessions } from "@/lib/cloud-agents-client";
import type { CloudAgentConversationMessage, CloudAgentRun, CloudAgentSession } from "@/types/cloud-agents";
import { useRouter } from "next/navigation";

interface ThreePaneProps {
  agentId: string;
  sessionId: string;
  accessToken: string | null;
}

export default function ThreePane({ agentId, sessionId, accessToken }: ThreePaneProps) {
  const router = useRouter();
  const [sessions, setSessions] = useState<CloudAgentSession[]>([]);
  const [sessionsLoading, setSessionsLoading] = useState(true);
  const [activeSession, setActiveSession] = useState<CloudAgentSession | null>(null);
  const [run, setRun] = useState<CloudAgentRun | null>(null);
  const [initialMessages, setInitialMessages] = useState<CloudAgentConversationMessage[]>([]);
  const [snapshotLoading, setSnapshotLoading] = useState(true);
  const [showNewSession, setShowNewSession] = useState(false);

  // Sessions sidebar
  useEffect(() => {
    if (!accessToken) return;
    let cancelled = false;
    setSessionsLoading(true);
    listCloudSessions(accessToken, agentId)
      .then((list) => {
        if (cancelled) return;
        setSessions(list);
        setActiveSession(list.find((s) => s.session_id === sessionId) ?? null);
      })
      .catch((e: unknown) => {
        if (!cancelled) {
          const msg = e instanceof Error ? e.message : String(e);
          message.error(`Failed to load sessions: ${msg}`);
        }
      })
      .finally(() => {
        if (!cancelled) setSessionsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [accessToken, agentId, sessionId]);

  // Conversation snapshot + active run
  useEffect(() => {
    if (!accessToken) return;
    let cancelled = false;
    setSnapshotLoading(true);
    Promise.all([
      getSessionConversation(accessToken, sessionId),
      activeSession?.active_run_id ? getCloudRun(accessToken, activeSession.active_run_id) : Promise.resolve(null),
    ])
      .then(([msgs, r]) => {
        if (cancelled) return;
        setInitialMessages(msgs);
        setRun(r);
      })
      .catch((e: unknown) => {
        if (!cancelled) {
          const msg = e instanceof Error ? e.message : String(e);
          message.error(`Failed to load conversation: ${msg}`);
        }
      })
      .finally(() => {
        if (!cancelled) setSnapshotLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [accessToken, sessionId, activeSession?.active_run_id]);

  const { events } = useSessionEventStream({
    sessionId,
    runId: activeSession?.active_run_id ?? null,
    enabled: Boolean(activeSession?.active_run_id),
  });

  return (
    <div className="flex h-[calc(100vh-64px)] w-full" data-testid="three-pane">
      <aside className="w-[280px] shrink-0 border-r border-gray-200">
        <SessionList
          sessions={sessions}
          loading={sessionsLoading}
          activeSessionId={sessionId}
          onNewSession={() => setShowNewSession(true)}
        />
      </aside>
      <main className="min-w-0 flex-1 border-r border-gray-200">
        {snapshotLoading ? (
          <div className="flex h-full items-center justify-center">
            <Spin />
          </div>
        ) : (
          <Conversation
            sessionId={sessionId}
            accessToken={accessToken}
            initialMessages={initialMessages}
            events={events}
          />
        )}
      </main>
      <aside className="w-[420px] shrink-0">
        <RightPanel run={run} events={events} />
      </aside>
      <NewSessionDialog
        open={showNewSession}
        agentId={agentId}
        accessToken={accessToken}
        onClose={() => setShowNewSession(false)}
        onCreated={(s) => {
          setShowNewSession(false);
          router.push(`/agents/${agentId}/sessions/${s.session_id}`);
        }}
      />
    </div>
  );
}
