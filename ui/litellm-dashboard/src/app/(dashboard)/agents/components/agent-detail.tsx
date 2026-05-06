"use client";

/**
 * AgentDetail — /agents/{agent_id} landing view.
 *
 * Renders the Agent's identity card, a settings hand-off link to
 * /settings/cloud-agents/ (G's territory), and the SessionList for
 * this agent. + New Session opens the create dialog.
 */
import { useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Card, Empty, Tag, Typography, Button, Space, message } from "antd";
import { relativeOrAbsolute } from "@/app/(dashboard)/agents/components/_dayjs";
import SessionList from "@/app/(dashboard)/agents/components/session-list";
import NewSessionDialog from "@/app/(dashboard)/agents/components/new-session-dialog";
import { getCloudAgent, listCloudSessions } from "@/lib/cloud-agents-client";
import type { CloudAgent, CloudAgentSession } from "@/types/cloud-agents";

const { Title, Paragraph, Text } = Typography;

interface AgentDetailProps {
  agentId: string;
  accessToken: string | null;
}

export default function AgentDetail({ agentId, accessToken }: AgentDetailProps) {
  const router = useRouter();
  const [agent, setAgent] = useState<CloudAgent | null>(null);
  const [sessions, setSessions] = useState<CloudAgentSession[]>([]);
  const [loading, setLoading] = useState(true);
  const [showNewSession, setShowNewSession] = useState(false);

  useEffect(() => {
    if (!accessToken) return;
    let cancelled = false;
    setLoading(true);
    Promise.all([getCloudAgent(accessToken, agentId), listCloudSessions(accessToken, agentId)])
      .then(([a, s]) => {
        if (cancelled) return;
        setAgent(a);
        setSessions(s);
      })
      .catch((e: unknown) => {
        if (!cancelled) {
          const msg = e instanceof Error ? e.message : String(e);
          message.error(`Failed to load agent: ${msg}`);
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [accessToken, agentId]);

  return (
    <div className="w-full p-6" data-testid="agent-detail">
      <Space direction="vertical" size="large" className="w-full">
        <div className="flex items-start justify-between gap-4">
          <div className="min-w-0">
            <Link href="/agents" className="!text-xs">
              ← All agents
            </Link>
            <Title level={3} className="!mb-0 !mt-1">
              {agent?.name ?? "Agent"}
            </Title>
            <Space size="small" className="mt-1">
              {agent?.model && <Tag color="blue">{agent.model}</Tag>}
              {agent?.last_activity_at && (
                <Text type="secondary" className="!text-xs">
                  Last activity {relativeOrAbsolute(agent.last_activity_at)}
                </Text>
              )}
            </Space>
          </div>
          <Space>
            <Link href="/settings/cloud-agents/" data-testid="agent-settings-link">
              <Button>Configure VM provider</Button>
            </Link>
            <Button type="primary" onClick={() => setShowNewSession(true)} data-testid="agent-detail-new-session-btn">
              + New Session
            </Button>
          </Space>
        </div>

        {agent?.system_prompt && (
          <Card size="small" data-testid="agent-system-prompt">
            <Text strong>System prompt</Text>
            <Paragraph className="!mb-0 !mt-1 whitespace-pre-wrap !text-sm">{agent.system_prompt}</Paragraph>
          </Card>
        )}

        <Card title="Sessions" data-testid="agent-detail-sessions">
          {sessions.length === 0 && !loading ? (
            <Empty description="No sessions yet — start one to spin up a VM." />
          ) : (
            <div className="-mx-6 -my-2">
              <SessionList
                sessions={sessions}
                loading={loading}
                activeSessionId={null}
                onNewSession={() => setShowNewSession(true)}
              />
            </div>
          )}
        </Card>
      </Space>

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
