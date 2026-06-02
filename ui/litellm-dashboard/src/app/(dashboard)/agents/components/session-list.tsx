"use client";

/**
 * SessionList — left sidebar showing all sessions belonging to the active
 * Agent. Used in both the per-agent landing page and the three-pane view.
 *
 * Rendered as a vertical list of SessionRow with a "+ New Session" action
 * pinned at the top.
 */
import { Button, Empty, Spin, Typography } from "antd";
import SessionRow from "@/app/(dashboard)/agents/components/session-row";
import type { CloudAgentSession } from "@/types/cloud-agents";

const { Text } = Typography;

interface SessionListProps {
  sessions: CloudAgentSession[];
  loading: boolean;
  activeSessionId?: string | null;
  onNewSession: () => void;
}

export default function SessionList({ sessions, loading, activeSessionId, onNewSession }: SessionListProps) {
  return (
    <div className="flex h-full flex-col" data-testid="session-list">
      <div className="flex items-center justify-between border-b border-gray-200 px-3 py-2">
        <Text strong>Sessions</Text>
        <Button size="small" type="primary" onClick={onNewSession} data-testid="new-session-btn">
          + New
        </Button>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto">
        {loading ? (
          <div className="flex items-center justify-center p-4">
            <Spin size="small" />
          </div>
        ) : sessions.length === 0 ? (
          <Empty description="No sessions yet" className="!my-8" />
        ) : (
          sessions.map((s) => <SessionRow key={s.session_id} session={s} active={s.session_id === activeSessionId} />)
        )}
      </div>
    </div>
  );
}
