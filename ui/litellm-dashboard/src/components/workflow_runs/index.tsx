import React, { useState, useEffect, useCallback } from "react";
import { Button, Collapse, Drawer, Empty, Spin, Table, Tooltip } from "antd";
import { ReloadOutlined } from "@ant-design/icons";
import { proxyBaseUrl } from "@/components/networking";
import { useTheme } from "@/contexts/ThemeContext";

interface WorkflowRunsProps {
  accessToken: string | null;
}

type RunStatus = "pending" | "running" | "paused" | "completed" | "failed";

interface RunMetadata {
  title?: string;
  state?: string;
  pr_url?: string;
  worktree_path?: string;
  plan_text?: string;
  grill_session_id?: string;
  session_id?: string;
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

// Status dot colors read fine on both light and dark surfaces, so they are shared.
const STATUS_DOT: Record<RunStatus, string> = {
  pending:   "#a1a1aa",
  running:   "#3b82f6",
  paused:    "#f59e0b",
  completed: "#22c55e",
  failed:    "#ef4444",
};

// Semantic palette - switches with the active theme. Everything in this view is
// inline-styled (so the global dark-mode CSS can't reach it); these tokens are
// the single source of truth for the colors used below.
interface Palette {
  pageBg: string;
  chipBg: string;
  border: string;
  borderSubtle: string;
  borderStrong: string;
  textStrong: string;
  textBody: string;
  textMuted: string;
  textFaint: string;
  barBg: string;
  barBorder: string;
  barText: string;
  link: string;
}

function palette(dark: boolean): Palette {
  return dark
    ? {
        pageBg: "transparent", // inherit the themed dashboard content area
        chipBg: "#1f1f1f",
        border: "#242424",
        borderSubtle: "#1e1e1e",
        borderStrong: "#282828",
        textStrong: "#e5e7eb",
        textBody: "#d4d4d8",
        textMuted: "#a1a1aa",
        textFaint: "#71717a",
        barBg: "#1a1a1a",
        barBorder: "#3f3f46",
        barText: "#a1a1aa",
        link: "#60a5fa",
      }
    : {
        pageBg: "#fff",
        chipBg: "#f4f4f5",
        border: "#e4e4e7",
        borderSubtle: "#f4f4f5",
        borderStrong: "#d4d4d8",
        textStrong: "#18181b",
        textBody: "#27272a",
        textMuted: "#52525b",
        textFaint: "#a1a1aa",
        barBg: "#f4f4f5",
        barBorder: "#d4d4d8",
        barText: "#71717a",
        link: "#2563eb",
      };
}

interface EventStyle {
  bar: string;
  border: string;
  text: string;
}

const EVENT_COLOR_LIGHT: Record<string, EventStyle> = {
  "step.started":  { bar: "#f0fdf4", border: "#86efac", text: "#16a34a" },
  "step.failed":   { bar: "#fef2f2", border: "#fca5a5", text: "#dc2626" },
  "hook.waiting":  { bar: "#fffbeb", border: "#fcd34d", text: "#d97706" },
  "hook.received": { bar: "#eff6ff", border: "#93c5fd", text: "#2563eb" },
};

const EVENT_COLOR_DARK: Record<string, EventStyle> = {
  "step.started":  { bar: "#16221a", border: "#2f6b3f", text: "#4ade80" },
  "step.failed":   { bar: "#241818", border: "#7f3a3a", text: "#f87171" },
  "hook.waiting":  { bar: "#241f12", border: "#7a5a1f", text: "#fbbf24" },
  "hook.received": { bar: "#161d2b", border: "#2f4f7a", text: "#60a5fa" },
};

function eventStyle(type: string, dark: boolean): EventStyle {
  if (dark) {
    return EVENT_COLOR_DARK[type] ?? { bar: "#1a1a1a", border: "#3f3f46", text: "#a1a1aa" };
  }
  return EVENT_COLOR_LIGHT[type] ?? { bar: "#f4f4f5", border: "#d4d4d8", text: "#52525b" };
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

// ── truncated text value ──────────────────────────────────────────────────────

const TRUNCATE_AT = 120;

const TruncatedValue: React.FC<{ value: string; c: Palette }> = ({ value, c }) => {
  const [expanded, setExpanded] = useState(false);
  if (value.length <= TRUNCATE_AT) {
    return <span style={{ color: c.textBody, wordBreak: "break-all" }}>{value}</span>;
  }
  return (
    <span style={{ color: c.textBody, wordBreak: "break-all" }}>
      {expanded ? value : value.slice(0, TRUNCATE_AT) + "…"}
      <button
        onClick={() => setExpanded((e) => !e)}
        style={{
          background: "none",
          border: "none",
          padding: "0 4px",
          cursor: "pointer",
          color: c.link,
          fontSize: 11,
          flexShrink: 0,
        }}
      >
        {expanded ? "less" : "more"}
      </button>
    </span>
  );
};

// ── metadata card ─────────────────────────────────────────────────────────────

const MetadataCard: React.FC<{ run: WorkflowRun; c: Palette }> = ({ run, c }) => {
  const meta = run.metadata ?? {};

  const primaryFields: { key: string; label: string }[] = [
    { key: "state",            label: "state" },
    { key: "worktree_path",    label: "worktree" },
    { key: "grill_session_id", label: "grill session" },
    { key: "session_id",       label: "session" },
  ];

  const primaryKeys = new Set(["title", ...primaryFields.map((f) => f.key)]);
  const extraEntries = Object.entries(meta).filter(
    ([k, v]) => !primaryKeys.has(k) && v !== null && v !== undefined && v !== ""
  );

  return (
    <div
      style={{
        borderRadius: 8,
        border: `1px solid ${c.border}`,
        marginBottom: 16,
        overflow: "hidden",
      }}
    >
      {/* title bar */}
      <div
        style={{
          padding: "14px 20px",
          borderBottom: `1px solid ${c.borderSubtle}`,
          display: "flex",
          alignItems: "center",
          gap: 10,
        }}
      >
        <StatusDot status={run.status} size={10} />
        <span style={{ fontSize: 14, fontWeight: 600, color: c.textStrong, flex: 1 }}>
          {runTitle(run)}
        </span>
        <span
          style={{
            fontFamily: "monospace",
            fontSize: 11,
            color: c.textFaint,
            background: c.chipBg,
            padding: "2px 8px",
            borderRadius: 4,
          }}
        >
          {shortId(run.run_id)}
        </span>
        <span
          style={{
            fontSize: 11,
            color: c.textFaint,
            background: c.chipBg,
            padding: "2px 8px",
            borderRadius: 4,
          }}
        >
          {run.workflow_type}
        </span>
      </div>

      {/* key fields grid */}
      <div
        style={{
          padding: "12px 20px",
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))",
          gap: "8px 24px",
          fontFamily: "monospace",
          fontSize: 12,
        }}
      >
        <FieldPair label="status" c={c}>
          <span style={{ textTransform: "capitalize", color: c.textBody }}>{run.status}</span>
        </FieldPair>
        <FieldPair label="created" c={c}>
          <span style={{ color: c.textBody }}>{timeAgo(run.created_at)}</span>
        </FieldPair>

        {meta.pr_url && (
          <FieldPair label="pr" c={c}>
            <a
              href={String(meta.pr_url)}
              target="_blank"
              rel="noopener noreferrer"
              style={{ color: c.link, textDecoration: "none", wordBreak: "break-all" }}
            >
              {String(meta.pr_url)}
            </a>
          </FieldPair>
        )}

        {primaryFields.map(({ key, label }) => {
          const v = meta[key];
          if (v === null || v === undefined || v === "") return null;
          const str = typeof v === "object" ? JSON.stringify(v) : String(v);
          return (
            <FieldPair key={key} label={label} c={c}>
              <TruncatedValue value={str} c={c} />
            </FieldPair>
          );
        })}

        {extraEntries.map(([k, v]) => {
          const str = typeof v === "object" ? JSON.stringify(v) : String(v);
          return (
            <FieldPair key={k} label={k} c={c}>
              <TruncatedValue value={str} c={c} />
            </FieldPair>
          );
        })}
      </div>
    </div>
  );
};

const FieldPair: React.FC<{ label: string; c: Palette; children: React.ReactNode }> = ({
  label,
  c,
  children,
}) => (
  <div style={{ display: "flex", flexDirection: "column", gap: 1 }}>
    <span style={{ fontSize: 10, color: c.textFaint, textTransform: "uppercase", letterSpacing: "0.06em" }}>
      {label}
    </span>
    <span style={{ fontSize: 12 }}>{children}</span>
  </div>
);

// ── gantt timeline ────────────────────────────────────────────────────────────

const GanttTimeline: React.FC<{
  run: WorkflowRun;
  events: WorkflowRunEvent[];
  c: Palette;
  dark: boolean;
}> = ({ run, events, c, dark }) => {
  if (events.length === 0) {
    return (
      <div style={{ padding: "16px 0", color: c.textFaint, fontSize: 12, fontFamily: "monospace" }}>
        No events recorded
      </div>
    );
  }

  const runStart = new Date(run.created_at).getTime();
  const eventTimes = events.map((e) => new Date(e.created_at).getTime());
  const lastTime = Math.max(...eventTimes);
  const totalSpan = Math.max(lastTime - runStart, 1);
  const totalDur = fmtDuration(lastTime - runStart);

  return (
    <div style={{ fontFamily: "monospace", fontSize: 12 }}>
      {/* ruler */}
      <div style={{ display: "grid", gridTemplateColumns: "160px 1fr", gap: "0 12px", marginBottom: 2 }}>
        <div />
        <div style={{ position: "relative", height: 16 }}>
          {[0, 100].map((pct) => (
            <span
              key={pct}
              style={{
                position: "absolute",
                left: `${pct}%`,
                transform: pct === 100 ? "translateX(-100%)" : undefined,
                fontSize: 10,
                color: c.textFaint,
              }}
            >
              {pct === 0 ? "0" : totalDur}
            </span>
          ))}
        </div>
      </div>

      {/* outer run bar */}
      <div style={{ display: "grid", gridTemplateColumns: "160px 1fr", gap: "0 12px", marginBottom: 4 }}>
        <div style={{ color: c.textMuted, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", paddingTop: 2 }}>
          {runTitle(run)}
        </div>
        <div
          style={{
            height: 24,
            background: c.barBg,
            border: `1px solid ${c.borderStrong}`,
            borderRadius: 4,
            display: "flex",
            alignItems: "center",
            paddingLeft: 8,
          }}
        >
          <span style={{ color: c.textMuted, fontSize: 11 }}>{totalDur}</span>
        </div>
      </div>

      {/* event rows */}
      <div style={{ display: "grid", gridTemplateColumns: "160px 1fr", gap: "0 12px", rowGap: 3 }}>
        {events.map((ev) => {
          const evTime = new Date(ev.created_at).getTime();
          const leftPct = ((evTime - runStart) / totalSpan) * 100;

          const nextIdx = events.findIndex((e) => e.sequence_number > ev.sequence_number);
          const nextTime =
            nextIdx >= 0
              ? new Date(events[nextIdx].created_at).getTime()
              : lastTime + Math.max(totalSpan * 0.12, 500);
          const widthPct = Math.max(8, ((nextTime - evTime) / totalSpan) * 100);
          const style = eventStyle(ev.event_type, dark);
          const dur = fmtDuration(nextTime - evTime);

          return (
            <React.Fragment key={ev.event_id}>
              <div
                style={{
                  color: style.text,
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                  paddingTop: 2,
                  paddingLeft: 12,
                }}
              >
                {ev.step_name || ev.event_type}
              </div>
              <div style={{ position: "relative", height: 24 }}>
                <Tooltip
                  title={
                    <div style={{ fontFamily: "monospace", fontSize: 11, lineHeight: 1.6 }}>
                      <div><span style={{ color: "#a1a1aa" }}>type: </span><span style={{ color: style.text }}>{ev.event_type}</span></div>
                      <div><span style={{ color: "#a1a1aa" }}>step: </span>{ev.step_name}</div>
                      <div><span style={{ color: "#a1a1aa" }}>seq:  </span>{ev.sequence_number}</div>
                      <div><span style={{ color: "#a1a1aa" }}>time: </span>{timeAgo(ev.created_at)}</div>
                      {ev.data && Object.keys(ev.data).length > 0 && (
                        <div><span style={{ color: "#a1a1aa" }}>data: </span>{JSON.stringify(ev.data)}</div>
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
                    <span style={{ color: style.text, whiteSpace: "nowrap", fontSize: 11 }}>{ev.event_type}</span>
                    {dur && <span style={{ color: c.textFaint, whiteSpace: "nowrap", fontSize: 11 }}>{dur}</span>}
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

// ── message row ───────────────────────────────────────────────────────────────

const MessageRow: React.FC<{ msg: WorkflowRunMessage; c: Palette }> = ({ msg, c }) => {
  const roleColor: Record<string, string> = {
    user:        "#2563eb",
    assistant:   "#16a34a",
    system:      "#7c3aed",
    tool_result: "#d97706",
  };
  const color = roleColor[msg.role] ?? "#52525b";

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "80px 1fr",
        gap: "0 16px",
        padding: "10px 0",
        borderBottom: `1px solid ${c.borderSubtle}`,
        fontFamily: "monospace",
        fontSize: 12,
        alignItems: "start",
      }}
    >
      <span style={{ color, paddingTop: 1 }}>[{msg.role}]</span>
      <div>
        <span
          style={{
            color: c.textBody,
            lineHeight: 1.6,
            whiteSpace: "pre-wrap",
            wordBreak: "break-word",
            display: "block",
          }}
        >
          {msg.content}
        </span>
        <span style={{ color: c.textFaint, fontSize: 11, marginTop: 2, display: "block" }}>
          {timeAgo(msg.created_at)}
        </span>
      </div>
    </div>
  );
};

// ── main component ────────────────────────────────────────────────────────────

const WorkflowRuns: React.FC<WorkflowRunsProps> = ({ accessToken }) => {
  const { isDarkMode } = useTheme();
  const c = palette(isDarkMode);

  const [runs, setRuns] = useState<WorkflowRun[]>([]);
  const [loadingRuns, setLoadingRuns] = useState(false);
  const [selectedRun, setSelectedRun] = useState<WorkflowRun | null>(null);
  const [events, setEvents] = useState<WorkflowRunEvent[]>([]);
  const [messages, setMessages] = useState<WorkflowRunMessage[]>([]);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);

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
      setDrawerOpen(true);
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
            (a: WorkflowRunEvent, b: WorkflowRunEvent) => a.sequence_number - b.sequence_number
          )
        );
        setMessages(
          [...(msgData.messages ?? [])].sort(
            (a: WorkflowRunMessage, b: WorkflowRunMessage) => a.sequence_number - b.sequence_number
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

  const columns = [
    {
      title: "Run",
      dataIndex: "run_id",
      key: "run",
      render: (_: string, run: WorkflowRun) => (
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <StatusDot status={run.status} size={7} />
          <div>
            <div style={{ fontSize: 13, color: c.textStrong, fontWeight: 500, lineHeight: 1.4 }}>
              {runTitle(run)}
            </div>
            <div style={{ fontFamily: "monospace", fontSize: 11, color: c.textFaint }}>
              {shortId(run.run_id)}
            </div>
          </div>
        </div>
      ),
    },
    {
      title: "Type",
      dataIndex: "workflow_type",
      key: "workflow_type",
      render: (v: string) => (
        <span style={{ fontFamily: "monospace", fontSize: 12, color: c.textMuted }}>{v}</span>
      ),
    },
    {
      title: "Status",
      dataIndex: "status",
      key: "status",
      render: (status: RunStatus, run: WorkflowRun) => {
        const state = run.metadata?.state;
        return (
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <StatusDot status={status} size={7} />
            <span style={{ fontSize: 12, color: c.textMuted, textTransform: "capitalize" }}>
              {state ?? status}
            </span>
          </div>
        );
      },
    },
    {
      title: "Created",
      dataIndex: "created_at",
      key: "created_at",
      render: (v: string) => (
        <span style={{ fontSize: 12, color: c.textFaint }}>{timeAgo(v)}</span>
      ),
    },
  ];

  return (
    <div
      style={{
        padding: "24px 32px",
        fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
        minHeight: "calc(100vh - 64px)",
        background: c.pageBg,
      }}
    >
      {/* page header */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          marginBottom: 20,
        }}
      >
        <div>
          <div style={{ fontSize: 18, fontWeight: 600, color: c.textStrong }}>Workflow Runs</div>
          <div style={{ fontSize: 13, color: c.textMuted, marginTop: 2 }}>
            Durable state tracking for agents and automated workflows
          </div>
        </div>
        <Button
          icon={<ReloadOutlined />}
          onClick={fetchRuns}
          loading={loadingRuns}
        >
          Refresh
        </Button>
      </div>

      {/* runs table — matches logs page density */}
      <div className="rounded-lg custom-border overflow-x-auto w-full">
        <Table
          dataSource={runs}
          columns={columns}
          rowKey="run_id"
          loading={loadingRuns}
          size="small"
          pagination={{ pageSize: 50, hideOnSinglePage: true, size: "small" }}
          onRow={(run) => ({
            onClick: () => fetchRunDetail(run),
            style: { cursor: "pointer" },
          })}
          locale={{
            emptyText: (
              <Empty
                description={<span style={{ color: c.textFaint, fontSize: 13 }}>No workflow runs yet</span>}
                image={Empty.PRESENTED_IMAGE_SIMPLE}
              />
            ),
          }}
          className="[&_.ant-table-cell]:py-0.5 [&_.ant-table-thead_.ant-table-cell]:py-1"
          style={{ border: "none" }}
        />
      </div>

      {/* detail drawer */}
      <Drawer
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        width={680}
        title={null}
        closable={false}
        bodyStyle={{ padding: 0 }}
        styles={{ body: { padding: 0 } }}
      >
        {!selectedRun ? null : loadingDetail ? (
          <div style={{ display: "flex", justifyContent: "center", padding: 80 }}>
            <Spin />
          </div>
        ) : (
          <div style={{ padding: "24px 28px", fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif' }}>
            {/* drawer close + refresh */}
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                marginBottom: 16,
              }}
            >
              <button
                onClick={() => setDrawerOpen(false)}
                style={{
                  background: "none",
                  border: "none",
                  cursor: "pointer",
                  padding: "4px 0",
                  fontSize: 12,
                  color: c.textFaint,
                  display: "flex",
                  alignItems: "center",
                  gap: 4,
                }}
              >
                ← close
              </button>
              <Button
                size="small"
                icon={<ReloadOutlined />}
                onClick={() => fetchRunDetail(selectedRun)}
                loading={loadingDetail}
              >
                Refresh
              </Button>
            </div>

            {/* metadata card — top */}
            <MetadataCard run={selectedRun} c={c} />

            {/* collapsible sections */}
            <Collapse
              defaultActiveKey={["timeline"]}
              ghost={false}
              style={{ border: `1px solid ${c.border}`, borderRadius: 8, overflow: "hidden" }}
              items={[
                {
                  key: "timeline",
                  label: (
                    <span style={{ fontSize: 12, fontWeight: 500, color: c.textMuted }}>
                      Timeline
                      <span style={{ marginLeft: 6, fontSize: 11, color: c.textFaint, fontWeight: 400 }}>
                        {events.length} {events.length === 1 ? "event" : "events"}
                      </span>
                    </span>
                  ),
                  children: (
                    <div style={{ padding: "4px 4px 12px" }}>
                      <GanttTimeline run={selectedRun} events={events} c={c} dark={isDarkMode} />
                    </div>
                  ),
                },
                {
                  key: "messages",
                  label: (
                    <span style={{ fontSize: 12, fontWeight: 500, color: c.textMuted }}>
                      Messages
                      <span style={{ marginLeft: 6, fontSize: 11, color: c.textFaint, fontWeight: 400 }}>
                        {messages.length}
                      </span>
                    </span>
                  ),
                  children: messages.length === 0 ? (
                    <div style={{ padding: "12px 4px", color: c.textFaint, fontSize: 12, fontFamily: "monospace" }}>
                      No messages
                    </div>
                  ) : (
                    <div style={{ paddingBottom: 4 }}>
                      {messages.map((msg) => (
                        <MessageRow key={msg.message_id} msg={msg} c={c} />
                      ))}
                    </div>
                  ),
                },
              ]}
            />
          </div>
        )}
      </Drawer>
    </div>
  );
};

export default WorkflowRuns;
