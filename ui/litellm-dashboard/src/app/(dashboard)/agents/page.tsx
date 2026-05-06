"use client";

/**
 * /agents — list of cloud-agent definitions (Cursor SDK on LiteLLM).
 *
 * Each row links to /agents/{agent_id} for the per-agent session list.
 * Definition-only — does not spin up a VM.
 */
import { useEffect, useState } from "react";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import AgentList from "@/app/(dashboard)/agents/components/agent-list";
import NewAgentDialog from "@/app/(dashboard)/agents/components/new-agent-dialog";
import { listCloudAgents } from "@/lib/cloud-agents-client";
import type { CloudAgent } from "@/types/cloud-agents";
import { Button, Typography, Space, message } from "antd";

const { Title, Paragraph } = Typography;

export default function CloudAgentsPage() {
  const { accessToken } = useAuthorized();
  const [agents, setAgents] = useState<CloudAgent[]>([]);
  const [loading, setLoading] = useState(true);
  const [showNewAgent, setShowNewAgent] = useState(false);

  const refresh = async () => {
    if (!accessToken) return;
    setLoading(true);
    try {
      const list = await listCloudAgents(accessToken);
      setAgents(list);
    } catch (e) {
      const errMsg = e instanceof Error ? e.message : String(e);
      message.error(`Failed to load agents: ${errMsg}`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [accessToken]);

  return (
    <div className="w-full p-8" data-testid="cloud-agents-page">
      <Space direction="vertical" size="large" className="w-full">
        <div className="flex items-center justify-between">
          <div>
            <Title level={3} className="!mb-1">
              Cloud Agents
            </Title>
            <Paragraph type="secondary" className="!mb-0">
              Long-running coding agents running on cloud VMs. Pick an agent to view its sessions.
            </Paragraph>
          </div>
          <Button type="primary" onClick={() => setShowNewAgent(true)} data-testid="new-agent-btn">
            + New Agent
          </Button>
        </div>
        <AgentList agents={agents} loading={loading} />
      </Space>
      <NewAgentDialog
        open={showNewAgent}
        onClose={() => setShowNewAgent(false)}
        onCreated={() => {
          setShowNewAgent(false);
          void refresh();
        }}
        accessToken={accessToken}
      />
    </div>
  );
}
