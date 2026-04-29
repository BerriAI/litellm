import React, { useState, useEffect, useCallback } from "react";
import { Button, Empty, Spin, Tooltip, Typography } from "antd";
import { ReloadOutlined } from "@ant-design/icons";
import { proxyBaseUrl } from "@/components/networking";

const { Text } = Typography;

interface WorkflowRunsProps {
  accessToken: string | null;
}

type RunStatus = "pending" | "running" | "paused" | "completed" | "failed";

interface RunMetadata {
  title?: string;
  state?: string;
  [key: string]: unknown;
}

interface WorkflowRun {
  run_id: string;
  status: RunStatus;
  workflow_type: string;
  created_at: string;
  metadata?: RunMetadata | null;
}

interface WorkflowRunEvent {
  event_id: string;
  event_type: string;
  step_name: string;
  sequence_number: number;
  created_at: string;
  data?: Record<string, unknown> | null;
}

interface WorkflowRunMessage {
  message_id: string;
  role: string;
  content: string;
  sequence_number: number;
  created_at: string;
}

// ── design tokens ─────────────────────────────────────────────────────────────

const STATUS_DOT: Record<RunStatus, string> = {
  pending: "#a1a1aa",
  running: "#3b82f6",
  paused: "#f59e0b",
  completed: "#22c55e",
  failed: "#ef4444",
};

const EVENT_COLOR: Record<string, { bar: string; border: string; text: string }> = {
  "step.started":  { bar: "#f0fdf4", border: "#86efac", text: "#16a34a" },
  "step.failed":   { bar: "#fef2f2", border: "#fca5a5", text: "#dc2626" },
  "hook.waiting":  { bar: "#fffbeb", border: "#fcd34d", text: "#d97706" },
  "hook.received": { bar: "#eff6ff", border: "#93c5fd", text: "#2563eb" },
};

function eventStyle(type: string) {
  return EVENT_COLOR[type] ?? { bar: "#f4f4f5", border: "#d4d4d8", text: "#52525b" };
}

// ── helpers ───────────────────────────────────────────────────────────────────

function timeAgo(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  if (isNaN(diff)) return iso;
  const s = Math.floor(diff / 1000);
  if (s < 60) return `${s}s ago`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}

function durationMs(from: string, to: string): number {
  return new Date(to).getTime() - new Date(from).getTime();
}

function fmtDuration(ms: number): string {
  if (ms < 0) return "";
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

function runTitle(run: WorkflowRun): string {
  const t = run.metadata?.title;
  if (t) return String(t);
  return run.workflow_type ?? run.run_id.slice(0, 8);
}

function shortId(id: string): string {
  return id.slice(0, 8);
}

// ── status dot ────────────────────────────────────────────────────────────────

const StatusDot: React.FC<{ status: RunStatus; size?: number }> = ({ status, size = 8 }) => (
  <span
    style={{
      display: "inline-block",
      width: size,
      height: size,
      borderRadius: "50%",
      background: STATUS_DOT[status] ?? "#a1a1aa",
      flexShrink: 0,
    }}
  />
);

// ── gantt timeline ────────────────────────────────────────────────────────────

const GanttTimeline: React.FC<{
  run: WorkflowRun;
  events: WorkflowRunEvent[];
}> = ({ run, events }) => {
  if (events.length === 0) {
    return (
      <div
        style={{
          padding: "20px 0",
          color: "#a1a1aa",
          fontSize: 12,
          fontFamily: "monospace",
        }}
      >
        No events recorded
      </div>
    );
  }

  const runStart = new Date(run.created_at).getTime();
  const eventTimes = events.map((e) => new Date(e.created_at).getTime());
  const lastTime = Math.max(...eventTimes);
  const totalSpan = Math.max(lastTime - runStart, 1);

  // outer bar duration label
  const totalDur = fmtDuration(lastTime - runStart);

  return (
    <div style={{ fontFamily: "monospace", fontSize: 12 }}>
      {/* ruler */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "180px 1fr",
          gap: "0 12px",
          marginBottom: 2,
        }}
      >
        <div />
        <div style={{ position: "relative", height: 16 }}>
          {[0, 25, 50, 75, 100].map((pct) => (
            <span
              key={pct}
              style={{
                position: "absolute",
                left: `${pct}%`,
                transform: "translateX(-50%)",
                fontSize: 10,
                color: "#a1a1aa",
              }}
            >
              {pct === 0 ? "0" : pct === 100 ? totalDur : ""}
            </span>
          ))}
        </div>
      </div>

      {/* outer run bar */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "180px 1fr",
          gap: "0 12px",
          marginBottom: 4,
        }}
      >
        <div
          style={{
            color: "#3f3f46",
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
            paddingTop: 2,
          }}
        >
          {runTitle(run)}
        </div>
        <div
          style={{
            position: "relative",
            height: 24,
            background: "#f4f4f5",
            border: "1px solid #d4d4d8",
            borderRadius: 4,
            display: "flex",
            alignItems: "center",
            paddingLeft: 8,
          }}
        >
          <span style={{ color: "#71717a", fontSize: 11 }}>{totalDur}</span>
        </div>
      </div>

      {/* vertical guide line */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "180px 1fr",
          gap: "0 12px",
          rowGap: 3,
        }}
      >
        {events.map((ev) => {
          const evTime = new Date(ev.created_at).getTime();
          const leftPct = ((evTime - runStart) / totalSpan) * 100;

          // width: time until next event or 8% minimum
          const nextIdx = events.findIndex((e) => e.sequence_number > ev.sequence_number);
          const nextTime =
            nextIdx >= 0
              ? new Date(events[nextIdx].created_at).getTime()
              : lastTime + Math.max(totalSpan * 0.12, 500);
          const widthPct = Math.max(
            8,
            ((nextTime - evTime) / totalSpan) * 100
          );

          const style = eventStyle(ev.event_type);
          const dur = fmtDuration(nextTime - evTime);

          return (
            <React.Fragment key={ev.event_id}>
              {/* label col */}
              <div
                style={{
                  color: "#52525b",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                  paddingTop: 2,
                  paddingLeft: 12,
                }}
              >
                <span style={{ color: style.text }}>{ev.step_name || ev.event_type}</span>
              </div>

              {/* bar col */}
              <div style={{ position: "relative", height: 24 }}>
                <Tooltip
                  title={
                    <div style={{ fontFamily: "monospace", fontSize: 11, lineHeight: 1.6 }}>
                      <div>
                        <span style={{ color: "#a1a1aa" }}>type: </span>
                        <span style={{ color: style.text }}>{ev.event_type}</span>
                      </div>
                      <div>
                        <span style={{ color: "#a1a1aa" }}>step: </span>
                        {ev.step_name}
                      </div>
                      <div>
                        <span style={{ color: "#a1a1aa" }}>seq:  </span>
                        {ev.sequence_number}
                      </div>
                      <div>
                        <span style={{ color: "#a1a1aa" }}>time: </span>
                        {timeAgo(ev.created_at)}
                      </div>
                      {ev.data && Object.keys(ev.data).length > 0 && (
                        <div>
                          <span style={{ color: "#a1a1aa" }}>data: </span>
                          {JSON.stringify(ev.data)}
                        </div>
                      )}
                    </div>
                  }
                >
                  <div
                    style={{
                      position: "absolute",
                      left: `${Math.min(leftPct, 92)}%`,
                      width: `${Math.min(widthPct, 100 - Math.min(leftPct, 92))}%`,
                      height: "100%",
                      background: style.bar,
                      border: `1px solid ${style.border}`,
                      borderRadius: 4,
                      display: "flex",
                      alignItems: "center",
                      paddingLeft: 8,
                      cursor: "default",
                      overflow: "hidden",
                      gap: 6,
                    }}
                  >
                    <span style={{ color: style.text, whiteSpace: "nowrap", fontSize: 11 }}>
                      {ev.event_type}
                    </span>
                    {dur && (
                      <span style={{ color: "#a1a1aa", whiteSpace: "nowrap", fontSize: 11 }}>
                        {dur}
                      </span>
                    )}
                  </div>
                </Tooltip>
              </div>
            </React.Fragment>
          );
        })}
      </div>
    </div>
  );
};

// ── messages list ─────────────────────────────────────────────────────────────

const MessageRow: React.FC<{ msg: WorkflowRunMessage }> = ({ msg }) => {
  const roleColor: Record<string, string> = {
    user: "#2563eb",
    assistant: "#16a34a",
    system: "#7c3aed",
    tool_result: "#d97706",
  };
  const color = roleColor[msg.role] ?? "#52525b";

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "80px 1fr",
        gap: "0 16px",
        paddingTop: 10,
        paddingBottom: 10,
        borderBottom: "1px solid #f4f4f5",
        fontFamily: "monospace",
        fontSize: 12,
        alignItems: "start",
      }}
    >
      <span style={{ color, paddingTop: 1 }}>[{msg.role}]</span>
      <div>
        <span
          style={{
            color: "#27272a",
            lineHeight: 1.6,
            whiteSpace: "pre-wrap",
            wordBreak: "break-word",
            display: "block",
            fontFamily: "inherit",
          }}
        >
          {msg.content}
        </span>
        <span style={{ color: "#a1a1aa", fontSize: 11, marginTop: 2, display: "block" }}>
          {timeAgo(msg.created_at)}
        </span>
      </div>
    </div>
  );
};

// ── main component ────────────────────────────────────────────────────────────

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
      const res = await fetch(`${proxyBaseUrl ?? ""}/v1/workflows/runs?limit=100`, {
        headers: { Authorization: `Bearer ${accessToken}` },
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setRuns(data.runs ?? []);
    } catch (err) {
      console.error("workflow runs fetch failed:", err);
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
        const [evRes, msgRes] = await Promise.all([
          fetch(`${base}/v1/workflows/runs/${run.run_id}/events`, {
            headers: { Authorization: `Bearer ${accessToken}` },
          }),
          fetch(`${base}/v1/workflows/runs/${run.run_id}/messages`, {
            headers: { Authorization: `Bearer ${accessToken}` },
          }),
        ]);
        const evData = evRes.ok ? await evRes.json() : { events: [] };
        const msgData = msgRes.ok ? await msgRes.json() : { messages: [] };
        setEvents(
          [...(evData.events ?? [])].sort(
            (a: WorkflowRunEvent, b: WorkflowRunEvent) =>
              a.sequence_number - b.sequence_number
          )
        );
        setMessages(
          [...(msgData.messages ?? [])].sort(
            (a: WorkflowRunMessage, b: WorkflowRunMessage) =>
              a.sequence_number - b.sequence_number
          )
        );
      } catch (err) {
        console.error("workflow run detail fetch failed:", err);
      } finally {
        setLoadingDetail(false);
      }
    },
    [accessToken]
  );

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
        fontFamily:
          '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
      }}
    >
      {/* ── left panel ── */}
      <div
        style={{
          width: 280,
          borderRight: "1px solid #e4e4e7",
          display: "flex",
          flexDirection: "column",
          overflow: "hidden",
          flexShrink: 0,
        }}
      >
        {/* panel header */}
        <div
          style={{
            padding: "12px 16px",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            borderBottom: "1px solid #e4e4e7",
          }}
        >
          <span style={{ fontSize: 13, fontWeight: 500, color: "#18181b" }}>
            Workflow Runs
            {runs.length > 0 && (
              <span
                style={{
                  marginLeft: 6,
                  fontSize: 11,
                  color: "#a1a1aa",
                  fontWeight: 400,
                }}
              >
                {runs.length}
              </span>
            )}
          </span>
          <button
            onClick={fetchRuns}
            disabled={loadingRuns}
            style={{
              background: "none",
              border: "none",
              cursor: "pointer",
              padding: 4,
              color: "#a1a1aa",
              display: "flex",
              alignItems: "center",
            }}
          >
            <ReloadOutlined style={{ fontSize: 13 }} spin={loadingRuns} />
          </button>
        </div>

        {/* run list */}
        <div style={{ flex: 1, overflowY: "auto" }}>
          {loadingRuns && runs.length === 0 ? (
            <div
              style={{ display: "flex", justifyContent: "center", padding: 40 }}
            >
              <Spin size="small" />
            </div>
          ) : runs.length === 0 ? (
            <Empty
              description={
                <span style={{ color: "#a1a1aa", fontSize: 12 }}>
                  No workflow runs
                </span>
              }
              style={{ marginTop: 48 }}
              image={Empty.PRESENTED_IMAGE_SIMPLE}
            />
          ) : (
            <div>
              {runs.map((run) => {
                const isSelected = selectedRun?.run_id === run.run_id;
                const label = runTitle(run);
                const state = run.metadata?.state;
                return (
                  <div
                    key={run.run_id}
                    onClick={() => fetchRunDetail(run)}
                    style={{
                      padding: "10px 16px",
                      cursor: "pointer",
                      background: isSelected ? "#fafafa" : "transparent",
                      borderLeft: isSelected
                        ? "2px solid #18181b"
                        : "2px solid transparent",
                      transition: "background 0.1s",
                    }}
                  >
                    {/* title */}
                    <div
                      style={{
                        fontSize: 12,
                        color: "#18181b",
                        lineHeight: 1.45,
                        marginBottom: 4,
                        overflow: "hidden",
                        display: "-webkit-box",
                        WebkitLineClamp: 2,
                        WebkitBoxOrient: "vertical",
                      }}
                    >
                      {label}
                    </div>
                    {/* meta row */}
                    <div
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: 6,
                      }}
                    >
                      <StatusDot status={run.status} size={7} />
                      <span
                        style={{
                          fontSize: 11,
                          color: "#71717a",
                          textTransform: "capitalize",
                        }}
                      >
                        {state ?? run.status}
                      </span>
                      <span style={{ color: "#d4d4d8", fontSize: 10 }}>·</span>
                      <span style={{ fontSize: 11, color: "#a1a1aa" }}>
                        {timeAgo(run.created_at)}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>

      {/* ── right panel ── */}
      <div style={{ flex: 1, overflowY: "auto" }}>
        {!selectedRun ? (
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              height: "100%",
              color: "#a1a1aa",
              fontSize: 13,
            }}
          >
            Select a run to view details
          </div>
        ) : loadingDetail ? (
          <div
            style={{
              display: "flex",
              justifyContent: "center",
              marginTop: 80,
            }}
          >
            <Spin />
          </div>
        ) : (
          <div style={{ padding: "24px 32px", maxWidth: 900 }}>
            {/* ── run header ── */}
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                marginBottom: 20,
              }}
            >
              {/* status + run id + meta */}
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 16,
                  flexWrap: "wrap",
                }}
              >
                {/* status pill */}
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 6,
                    background: "#f4f4f5",
                    border: "1px solid #e4e4e7",
                    borderRadius: 6,
                    padding: "3px 10px",
                  }}
                >
                  <StatusDot status={selectedRun.status} size={8} />
                  <span
                    style={{
                      fontSize: 12,
                      fontWeight: 500,
                      color: "#3f3f46",
                      textTransform: "capitalize",
                    }}
                  >
                    {selectedRun.status}
                  </span>
                </div>

                {/* run id */}
                <span
                  style={{
                    fontFamily: "monospace",
                    fontSize: 12,
                    color: "#71717a",
                  }}
                >
                  {shortId(selectedRun.run_id)}
                </span>

                {/* workflow type */}
                <span style={{ fontSize: 12, color: "#a1a1aa" }}>
                  {selectedRun.workflow_type}
                </span>

                {/* duration */}
                {events.length > 0 && (
                  <span
                    style={{
                      fontFamily: "monospace",
                      fontSize: 12,
                      color: "#a1a1aa",
                    }}
                  >
                    {fmtDuration(
                      durationMs(
                        selectedRun.created_at,
                        events[events.length - 1].created_at
                      )
                    )}
                  </span>
                )}
              </div>

              <Button
                size="small"
                icon={<ReloadOutlined />}
                onClick={() => fetchRunDetail(selectedRun)}
                loading={loadingDetail}
                style={{ color: "#71717a", borderColor: "#e4e4e7" }}
              >
                Refresh
              </Button>
            </div>

            {/* ── timeline ── */}
            <div
              style={{
                borderRadius: 8,
                border: "1px solid #e4e4e7",
                padding: "16px 20px",
                marginBottom: 24,
                background: "#fff",
              }}
            >
              <div
                style={{
                  fontSize: 11,
                  fontWeight: 500,
                  color: "#a1a1aa",
                  letterSpacing: "0.05em",
                  textTransform: "uppercase",
                  marginBottom: 12,
                }}
              >
                Timeline · {events.length} events
              </div>
              <GanttTimeline run={selectedRun} events={events} />
            </div>

            {/* ── messages ── */}
            {messages.length > 0 && (
              <div
                style={{
                  borderRadius: 8,
                  border: "1px solid #e4e4e7",
                  padding: "16px 20px",
                  background: "#fff",
                }}
              >
                <div
                  style={{
                    fontSize: 11,
                    fontWeight: 500,
                    color: "#a1a1aa",
                    letterSpacing: "0.05em",
                    textTransform: "uppercase",
                    marginBottom: 4,
                  }}
                >
                  Messages · {messages.length}
                </div>
                <div>
                  {messages.map((msg) => (
                    <MessageRow key={msg.message_id} msg={msg} />
                  ))}
                </div>
              </div>
            )}

            {/* ── run metadata (collapsed details) ── */}
            {selectedRun.metadata && Object.keys(selectedRun.metadata).length > 0 && (
              <div
                style={{
                  marginTop: 16,
                  borderRadius: 8,
                  border: "1px solid #e4e4e7",
                  padding: "16px 20px",
                }}
              >
                <div
                  style={{
                    fontSize: 11,
                    fontWeight: 500,
                    color: "#a1a1aa",
                    letterSpacing: "0.05em",
                    textTransform: "uppercase",
                    marginBottom: 10,
                  }}
                >
                  Metadata
                </div>
                <div style={{ fontFamily: "monospace", fontSize: 11 }}>
                  {Object.entries(selectedRun.metadata)
                    .filter(([, v]) => v !== null && v !== undefined && v !== "")
                    .map(([k, v]) => (
                      <div
                        key={k}
                        style={{
                          display: "grid",
                          gridTemplateColumns: "140px 1fr",
                          gap: "0 12px",
                          paddingTop: 4,
                          paddingBottom: 4,
                          borderBottom: "1px solid #f4f4f5",
                        }}
                      >
                        <span style={{ color: "#a1a1aa" }}>{k}</span>
                        <span
                          style={{
                            color: "#27272a",
                            wordBreak: "break-all",
                          }}
                        >
                          {typeof v === "object" ? JSON.stringify(v) : String(v)}
                        </span>
                      </div>
                    ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default WorkflowRuns;
