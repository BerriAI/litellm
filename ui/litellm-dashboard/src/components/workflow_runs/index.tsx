import React, { useState, useEffect, useCallback } from "react";
import {
  Button,
  List,
  Spin,
  Empty,
  Table,
  Tag,
  Typography,
  Divider,
} from "antd";
import { ReloadOutlined } from "@ant-design/icons";
import { proxyBaseUrl } from "@/components/networking";
import type { ColumnsType } from "antd/es/table";
const { Text, Title } = Typography;

interface WorkflowRunsProps {
  accessToken: string | null;
}

type WorkflowStatus = "pending" | "running" | "paused" | "completed" | "failed";

interface WorkflowRun {
  run_id: string;
  status: WorkflowStatus;
  workflow_type: string;
  created_at: string;
  first_message?: string;
}

interface WorkflowRunEvent {
  event_type: string;
  step_name: string;
  sequence_number: number;
  created_at: string;
}

interface WorkflowRunMessage {
  role: string;
  content: string;
  sequence_number: number;
}

interface WorkflowRunsListResponse {
  runs: WorkflowRun[];
  count: number;
}

interface WorkflowRunEventsResponse {
  events: WorkflowRunEvent[];
}

interface WorkflowRunMessagesResponse {
  messages: WorkflowRunMessage[];
}

function statusTagColor(
  status: WorkflowStatus
): "default" | "processing" | "warning" | "success" | "error" {
  switch (status) {
    case "pending":
      return "default";
    case "running":
      return "processing";
    case "paused":
      return "warning";
    case "completed":
      return "success";
    case "failed":
      return "error";
    default:
      return "default";
  }
}

function timeAgo(isoDate: string): string {
  try {
    const diff = Date.now() - new Date(isoDate).getTime();
    if (isNaN(diff)) return isoDate;
    const secs = Math.floor(diff / 1000);
    if (secs < 60) return `${secs}s ago`;
    const mins = Math.floor(secs / 60);
    if (mins < 60) return `${mins}m ago`;
    const hrs = Math.floor(mins / 60);
    if (hrs < 24) return `${hrs}h ago`;
    const days = Math.floor(hrs / 24);
    return `${days}d ago`;
  } catch {
    return isoDate;
  }
}

function truncateText(text: string, maxLen: number): string {
  if (!text) return "";
  if (text.length <= maxLen) return text;
  return text.slice(0, maxLen) + "…";
}

const eventColumns: ColumnsType<WorkflowRunEvent> = [
  {
    title: "TYPE",
    dataIndex: "event_type",
    key: "event_type",
    render: (val: string) => <Text code>{val}</Text>,
  },
  {
    title: "STEP",
    dataIndex: "step_name",
    key: "step_name",
    render: (val: string) => val ?? "-",
  },
  {
    title: "SEQ",
    dataIndex: "sequence_number",
    key: "sequence_number",
    width: 70,
  },
  {
    title: "TIME",
    dataIndex: "created_at",
    key: "created_at",
    render: (val: string) => timeAgo(val),
  },
];

const WorkflowRuns: React.FC<WorkflowRunsProps> = ({ accessToken }) => {
  const [runs, setRuns] = useState<WorkflowRun[]>([]);
  const [loadingRuns, setLoadingRuns] = useState(false);
  const [selectedRun, setSelectedRun] = useState<WorkflowRun | null>(null);
  const [events, setEvents] = useState<WorkflowRunEvent[]>([]);
  const [messages, setMessages] = useState<WorkflowRunMessage[]>([]);
  const [loadingDetail, setLoadingDetail] = useState(false);

  const fetchRuns = useCallback(async () => {
    if (!accessToken) return;
    setLoadingRuns(true);
    try {
      const base = proxyBaseUrl ?? "";
      const res = await fetch(`${base}/v1/workflows/runs`, {
        headers: { Authorization: `Bearer ${accessToken}` },
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: WorkflowRunsListResponse = await res.json();
      setRuns(data.runs ?? []);
    } catch (err) {
      console.error("Failed to fetch workflow runs:", err);
    } finally {
      setLoadingRuns(false);
    }
  }, [accessToken]);

  const fetchRunDetail = useCallback(
    async (run: WorkflowRun) => {
      if (!accessToken) return;
      setSelectedRun(run);
      setLoadingDetail(true);
      setEvents([]);
      setMessages([]);
      try {
        const base = proxyBaseUrl ?? "";
        const [eventsRes, messagesRes] = await Promise.all([
          fetch(`${base}/v1/workflows/runs/${run.run_id}/events`, {
            headers: { Authorization: `Bearer ${accessToken}` },
          }),
          fetch(`${base}/v1/workflows/runs/${run.run_id}/messages`, {
            headers: { Authorization: `Bearer ${accessToken}` },
          }),
        ]);
        const eventsData: WorkflowRunEventsResponse = eventsRes.ok
          ? await eventsRes.json()
          : { events: [] };
        const messagesData: WorkflowRunMessagesResponse = messagesRes.ok
          ? await messagesRes.json()
          : { messages: [] };
        const sortedEvents = [...(eventsData.events ?? [])].sort(
          (a, b) => a.sequence_number - b.sequence_number
        );
        const sortedMessages = [...(messagesData.messages ?? [])].sort(
          (a, b) => a.sequence_number - b.sequence_number
        );
        setEvents(sortedEvents);
        setMessages(sortedMessages);
      } catch (err) {
        console.error("Failed to fetch run detail:", err);
      } finally {
        setLoadingDetail(false);
      }
    },
    [accessToken]
  );

  const refreshDetail = useCallback(() => {
    if (selectedRun) fetchRunDetail(selectedRun);
  }, [selectedRun, fetchRunDetail]);

  useEffect(() => {
    fetchRuns();
  }, [fetchRuns]);

  return (
    <div
      style={{
        display: "flex",
        height: "calc(100vh - 64px)",
        background: "#fff",
        overflow: "hidden",
      }}
    >
      {/* Left panel — run list */}
      <div
        style={{
          width: 300,
          borderRight: "1px solid #f0f0f0",
          display: "flex",
          flexDirection: "column",
          overflow: "hidden",
        }}
      >
        <div
          style={{
            padding: "16px 16px 8px",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            borderBottom: "1px solid #f0f0f0",
          }}
        >
          <Title level={5} style={{ margin: 0 }}>
            Workflow Runs
          </Title>
          <Button
            size="small"
            icon={<ReloadOutlined />}
            onClick={fetchRuns}
            loading={loadingRuns}
          />
        </div>

        <div style={{ flex: 1, overflowY: "auto" }}>
          {loadingRuns && runs.length === 0 ? (
            <div
              style={{ display: "flex", justifyContent: "center", padding: 32 }}
            >
              <Spin />
            </div>
          ) : runs.length === 0 ? (
            <Empty
              description="No workflow runs"
              style={{ marginTop: 48 }}
              image={Empty.PRESENTED_IMAGE_SIMPLE}
            />
          ) : (
            <List
              dataSource={runs}
              renderItem={(run) => {
                const isSelected = selectedRun?.run_id === run.run_id;
                return (
                  <List.Item
                    key={run.run_id}
                    onClick={() => fetchRunDetail(run)}
                    style={{
                      padding: "10px 16px",
                      cursor: "pointer",
                      background: isSelected ? "#e6f4ff" : "transparent",
                      borderLeft: isSelected
                        ? "3px solid #1677ff"
                        : "3px solid transparent",
                      transition: "background 0.15s",
                    }}
                  >
                    <div style={{ width: "100%" }}>
                      <div
                        style={{
                          display: "flex",
                          justifyContent: "space-between",
                          alignItems: "center",
                          marginBottom: 4,
                        }}
                      >
                        <Text
                          style={{
                            fontFamily: "monospace",
                            fontSize: 12,
                            color: "#595959",
                          }}
                        >
                          {truncateText(run.run_id, 18)}
                        </Text>
                        <Tag color={statusTagColor(run.status)}>
                          {run.status}
                        </Tag>
                      </div>
                      <div
                        style={{
                          display: "flex",
                          justifyContent: "space-between",
                          alignItems: "center",
                        }}
                      >
                        <Text
                          type="secondary"
                          style={{ fontSize: 12 }}
                          ellipsis
                        >
                          {truncateText(
                            run.first_message ?? run.workflow_type ?? "",
                            28
                          )}
                        </Text>
                        <Text
                          type="secondary"
                          style={{ fontSize: 11, whiteSpace: "nowrap" }}
                        >
                          {timeAgo(run.created_at)}
                        </Text>
                      </div>
                    </div>
                  </List.Item>
                );
              }}
            />
          )}
        </div>
      </div>

      {/* Right panel — run detail */}
      <div style={{ flex: 1, overflowY: "auto", padding: "24px 32px" }}>
        {!selectedRun ? (
          <Empty
            description="Select a run to view details"
            style={{ marginTop: 80 }}
          />
        ) : loadingDetail ? (
          <div style={{ display: "flex", justifyContent: "center", marginTop: 80 }}>
            <Spin size="large" />
          </div>
        ) : (
          <>
            {/* Header */}
            <div
              style={{
                display: "flex",
                alignItems: "flex-start",
                justifyContent: "space-between",
                marginBottom: 24,
              }}
            >
              <div>
                <Text
                  style={{
                    fontFamily: "monospace",
                    fontSize: 15,
                    fontWeight: 600,
                    display: "block",
                    marginBottom: 8,
                  }}
                >
                  {selectedRun.run_id}
                </Text>
                <div
                  style={{ display: "flex", alignItems: "center", gap: 8 }}
                >
                  <Tag color={statusTagColor(selectedRun.status)}>
                    {selectedRun.status}
                  </Tag>
                  <Text type="secondary" style={{ fontSize: 12 }}>
                    {selectedRun.workflow_type}
                  </Text>
                  <Text type="secondary" style={{ fontSize: 12 }}>
                    ·
                  </Text>
                  <Text type="secondary" style={{ fontSize: 12 }}>
                    {timeAgo(selectedRun.created_at)}
                  </Text>
                </div>
              </div>
              <Button
                icon={<ReloadOutlined />}
                onClick={refreshDetail}
                loading={loadingDetail}
              >
                Refresh
              </Button>
            </div>

            <Divider orientation="left">Events</Divider>

            {/* Events table */}
            {events.length === 0 ? (
              <Empty
                description="No events"
                image={Empty.PRESENTED_IMAGE_SIMPLE}
                style={{ marginBottom: 24 }}
              />
            ) : (
              <Table
                dataSource={events}
                columns={eventColumns}
                rowKey={(r) => `${r.event_type}-${r.sequence_number}`}
                size="small"
                pagination={false}
                style={{ marginBottom: 24 }}
              />
            )}

            <Divider orientation="left">Messages</Divider>

            {/* Messages list */}
            {messages.length === 0 ? (
              <Empty
                description="No messages"
                image={Empty.PRESENTED_IMAGE_SIMPLE}
              />
            ) : (
              <List
                dataSource={messages}
                renderItem={(msg) => (
                  <List.Item
                    key={msg.sequence_number}
                    style={{ padding: "8px 0", alignItems: "flex-start" }}
                  >
                    <div>
                      <Tag
                        color={msg.role === "assistant" ? "blue" : "default"}
                        style={{ marginBottom: 4 }}
                      >
                        {msg.role}
                      </Tag>
                      <Text style={{ display: "block", whiteSpace: "pre-wrap" }}>
                        {msg.content}
                      </Text>
                    </div>
                  </List.Item>
                )}
              />
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default WorkflowRuns;
