"use client";

import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useParams, useRouter } from "next/navigation";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

type MessageStatus = "in_progress" | "completed" | "failed";
type MessageRole = "user" | "assistant";

interface ToolCall {
  name: string;
  input?: unknown;
  output?: string;
}

interface MessageRow {
  id: string;
  session_id: string;
  role: MessageRole;
  content: string;
  status: MessageStatus;
  created_at: string;
  completed_at?: string;
  tools?: ToolCall[];
  model?: string;
  error_reason?: string;
}

interface SandboxSpec {
  type: string;
  size: string;
  timeout_minutes?: number;
  idle_timeout_minutes?: number;
}

interface RepoSpec {
  url: string;
  starting_ref: string;
  checked_out_sha?: string;
}

interface SessionRow {
  id: string;
  agent_id: string;
  agent_name?: string;
  sandbox: SandboxSpec;
  status: string;
  repos: RepoSpec[];
  created_by: string;
  created_at: string;
  terminated_at: string | null;
  default_model?: string;
}

interface ListResponse<T> {
  data: T[];
  next_cursor: string | null;
  has_more: boolean;
}

const DEFAULT_PROXY = "http://localhost:4000";
const DEFAULT_KEY = "sk-1234";
const POLL_INTERVAL_MS = 2000;
const COLUMN_MAX_WIDTH = 720;

const getProxyBase = (): string =>
  (typeof window !== "undefined" &&
    localStorage.getItem("LITELLM_PROXY_URL")) ||
  DEFAULT_PROXY;

const getApiKey = (): string =>
  (typeof window !== "undefined" && localStorage.getItem("LITELLM_API_KEY")) ||
  DEFAULT_KEY;

const buildHeaders = (): HeadersInit => ({
  "Content-Type": "application/json",
  Authorization: `Bearer ${getApiKey()}`,
});

const truncate = (s: string, n: number): string =>
  s.length > n ? s.slice(0, n) + "…" : s;

const formatRelative = (iso: string | null | undefined): string => {
  if (!iso) return "";
  try {
    const d = new Date(iso).getTime();
    const now = Date.now();
    const diff = Math.max(0, now - d);
    const sec = Math.floor(diff / 1000);
    if (sec < 60) return `${sec}s`;
    const min = Math.floor(sec / 60);
    if (min < 60) return `${min}m`;
    const hr = Math.floor(min / 60);
    if (hr < 24) return `${hr}h`;
    const day = Math.floor(hr / 24);
    if (day < 7) return `${day}d`;
    const wk = Math.floor(day / 7);
    if (wk < 5) return `${wk}w`;
    const mo = Math.floor(day / 30);
    if (mo < 12) return `${mo}mo`;
    return `${Math.floor(day / 365)}y`;
  } catch {
    return "";
  }
};

const statusColor = (status: string): string => {
  switch ((status || "").toLowerCase()) {
    case "error":
    case "failed":
      return "#dc2626";
    case "provisioning":
    case "pending":
    case "queued":
      return "#d97706";
    default:
      // ready, terminated, anything else — subtle gray
      return "#bcbcc0";
  }
};

const groupSessionsByAge = (
  sessions: SessionRow[],
): Array<{ label: string; items: SessionRow[] }> => {
  const now = Date.now();
  const day = 1000 * 60 * 60 * 24;
  const buckets: Record<string, SessionRow[]> = {
    Today: [],
    "This Week": [],
    "Last Week": [],
    Older: [],
  };
  for (const s of sessions) {
    const age = now - new Date(s.created_at).getTime();
    if (age < day) buckets.Today.push(s);
    else if (age < 7 * day) buckets["This Week"].push(s);
    else if (age < 14 * day) buckets["Last Week"].push(s);
    else buckets.Older.push(s);
  }
  return [
    { label: "Today", items: buckets.Today },
    { label: "This Week", items: buckets["This Week"] },
    { label: "Last Week", items: buckets["Last Week"] },
    { label: "Older", items: buckets.Older },
  ].filter((g) => g.items.length > 0);
};

export default function SessionThreadView() {
  const params = useParams<{ sid: string }>();
  const router = useRouter();
  const sessionId = params?.sid || "";

  const [session, setSession] = useState<SessionRow | null>(null);
  const [messages, setMessages] = useState<MessageRow[]>([]);
  const [draft, setDraft] = useState<string>("");
  const [sending, setSending] = useState<boolean>(false);
  const [aborting, setAborting] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [sessionsList, setSessionsList] = useState<SessionRow[]>([]);
  const [agentNameById, setAgentNameById] = useState<Record<string, string>>(
    {},
  );

  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  // Fetch sessions + agents (for the left rail)
  useEffect(() => {
    let cancelled = false;
    (async () => {
      const proxy = getProxyBase();
      const headers = buildHeaders();
      try {
        const [sRes, aRes] = await Promise.all([
          fetch(`${proxy}/v2/sessions?limit=100`, { headers }),
          fetch(`${proxy}/v2/agents?limit=100`, { headers }),
        ]);
        if (cancelled) return;
        if (sRes.ok) {
          const data: ListResponse<SessionRow> = await sRes.json();
          setSessionsList(data.data || []);
        }
        if (aRes.ok) {
          const data: ListResponse<{ id: string; name: string }> =
            await aRes.json();
          const map: Record<string, string> = {};
          for (const a of data.data || []) map[a.id] = a.name;
          setAgentNameById(map);
        }
      } catch {
        // silent — rail is non-critical
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [sessionId]); // refresh rail when nav changes

  const hasInProgress = useMemo(
    () => messages.some((m) => m.status === "in_progress"),
    [messages],
  );

  const currentModel = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      const m = messages[i];
      if (m.role === "assistant" && m.model) return m.model;
    }
    return session?.default_model || "";
  }, [messages, session]);

  const loadSession = useCallback(async () => {
    if (!sessionId) return;
    setLoading(true);
    setError(null);
    try {
      const proxy = getProxyBase();
      const headers = buildHeaders();
      const [sessionRes, messagesRes] = await Promise.all([
        fetch(`${proxy}/v2/sessions/${sessionId}`, { headers }),
        fetch(`${proxy}/v2/sessions/${sessionId}/messages`, { headers }),
      ]);

      if (sessionRes.ok) {
        setSession(await sessionRes.json());
      } else {
        throw new Error(`Failed to fetch session: ${sessionRes.status}`);
      }

      if (messagesRes.ok) {
        const m: ListResponse<MessageRow> = await messagesRes.json();
        setMessages(m.data || []);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [sessionId]);

  useEffect(() => {
    loadSession();
  }, [loadSession]);

  // Poll while any message is in_progress
  useEffect(() => {
    if (!sessionId || !hasInProgress) return;
    let cancelled = false;
    const interval = setInterval(async () => {
      try {
        const res = await fetch(
          `${getProxyBase()}/v2/sessions/${sessionId}/messages`,
          { headers: buildHeaders() },
        );
        if (!res.ok || cancelled) return;
        const m: ListResponse<MessageRow> = await res.json();
        if (!cancelled) setMessages(m.data || []);
      } catch {
        // silent
      }
    }, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [sessionId, hasInProgress]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({
      behavior: "smooth",
      block: "end",
    });
  }, [messages]);

  const handleSend = useCallback(async () => {
    const content = draft.trim();
    if (!content || !sessionId || sending) return;
    setSending(true);
    setError(null);

    const optimisticId = `optimistic-${Date.now()}`;
    const optimistic: MessageRow = {
      id: optimisticId,
      session_id: sessionId,
      role: "user",
      content,
      status: "completed",
      created_at: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, optimistic]);
    setDraft("");

    try {
      const res = await fetch(
        `${getProxyBase()}/v2/sessions/${sessionId}/messages`,
        {
          method: "POST",
          headers: buildHeaders(),
          body: JSON.stringify({ content }),
        },
      );
      if (!res.ok) {
        const errText = await res.text().catch(() => "");
        throw new Error(`${res.status} ${errText || res.statusText}`);
      }
      const refreshed = await fetch(
        `${getProxyBase()}/v2/sessions/${sessionId}/messages`,
        { headers: buildHeaders() },
      );
      if (refreshed.ok) {
        const m: ListResponse<MessageRow> = await refreshed.json();
        setMessages(m.data || []);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setMessages((prev) => prev.filter((x) => x.id !== optimisticId));
    } finally {
      setSending(false);
    }
  }, [draft, sessionId, sending]);

  const handleAbort = useCallback(async () => {
    if (!sessionId || aborting) return;
    setAborting(true);
    setError(null);
    try {
      const res = await fetch(
        `${getProxyBase()}/v2/sessions/${sessionId}/abort`,
        { method: "POST", headers: buildHeaders() },
      );
      if (!res.ok) {
        const errText = await res.text().catch(() => "");
        throw new Error(`${res.status} ${errText || res.statusText}`);
      }
      const refreshed = await fetch(
        `${getProxyBase()}/v2/sessions/${sessionId}/messages`,
        { headers: buildHeaders() },
      );
      if (refreshed.ok) {
        const m: ListResponse<MessageRow> = await refreshed.json();
        setMessages(m.data || []);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setAborting(false);
    }
  }, [sessionId, aborting]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend],
  );

  return (
    <div style={{ display: "flex", height: "100vh" }}>
      <SessionsRail
        sessions={sessionsList}
        agentNameById={agentNameById}
        activeId={sessionId}
        onPick={(id) => router.push(`/sessions/${id}`)}
      />

      <div
        style={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          minWidth: 0,
        }}
      >
        <div className="sessions-header">
          <span
            style={{
              color: "var(--text-primary)",
              fontSize: 13,
              fontWeight: 500,
              letterSpacing: "-0.005em",
            }}
          >
            {session?.agent_name ||
              (session ? agentNameById[session.agent_id] : "") ||
              "Session"}
          </span>
          {session?.status && (
            <span
              style={{
                fontSize: 11,
                color: "var(--text-muted)",
                display: "inline-flex",
                alignItems: "center",
              }}
            >
              {session.status !== "ready" && (
                <span
                  className="sessions-status-dot"
                  style={{ background: statusColor(session.status) }}
                />
              )}
              {session.status}
            </span>
          )}
          <div className="sessions-header-right" />
        </div>

        <div style={{ flex: 1, overflowY: "auto" }}>
          <div
            style={{
              maxWidth: COLUMN_MAX_WIDTH,
              margin: "0 auto",
              padding: "32px 24px 24px",
            }}
          >
            {loading && messages.length === 0 && (
              <div style={{ color: "var(--text-muted)", fontSize: 13 }}>
                Loading…
              </div>
            )}
            {!loading && messages.length === 0 && (
              <div style={{ color: "var(--text-muted)", fontSize: 13 }}>
                No messages. Send one below.
              </div>
            )}
            {messages.map((m) => (
              <MessageRowView key={m.id} msg={m} />
            ))}
            <div ref={messagesEndRef} />
          </div>
        </div>

        <div style={{ background: "var(--bg-page)" }}>
          <div
            style={{
              maxWidth: COLUMN_MAX_WIDTH,
              margin: "0 auto",
              padding: "10px 24px 20px",
            }}
          >
            <div className="sessions-composer-wrap">
              <textarea
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Add a follow up"
                disabled={sending}
                rows={2}
                className="sessions-composer-textarea"
              />
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  marginTop: 8,
                  minHeight: 22,
                }}
              >
                <span
                  className="sessions-mono"
                  style={{
                    fontSize: 10.5,
                    color: error ? "#b91c1c" : "var(--text-muted)",
                  }}
                >
                  {error
                    ? error
                    : sending
                      ? "sending…"
                      : currentModel || ""}
                </span>
                {hasInProgress ? (
                  <button
                    className="sessions-stop-btn"
                    onClick={handleAbort}
                    disabled={aborting}
                    aria-label="Stop"
                    title="Stop"
                  >
                    <svg
                      width="9"
                      height="9"
                      viewBox="0 0 9 9"
                      fill="currentColor"
                    >
                      <rect x="0" y="0" width="9" height="9" rx="1" />
                    </svg>
                  </button>
                ) : (
                  <span
                    className="sessions-mono"
                    style={{
                      fontSize: 10.5,
                      color: "var(--text-muted)",
                    }}
                  >
                    ⌘+↵
                  </span>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function SessionsRail({
  sessions,
  agentNameById,
  activeId,
  onPick,
}: {
  sessions: SessionRow[];
  agentNameById: Record<string, string>;
  activeId: string;
  onPick: (id: string) => void;
}) {
  const groups = useMemo(() => groupSessionsByAge(sessions), [sessions]);
  return (
    <aside
      style={{
        width: 230,
        flexShrink: 0,
        borderRight: "1px solid var(--border-color)",
        background: "var(--bg-rail)",
        overflowY: "auto",
        height: "100vh",
      }}
    >
      <div
        style={{
          padding: "10px 14px",
          fontSize: 12.5,
          fontWeight: 600,
          color: "var(--text-primary)",
          letterSpacing: "-0.005em",
        }}
      >
        Sessions
      </div>
      {sessions.length === 0 && (
        <div
          style={{
            padding: "4px 14px",
            color: "var(--text-muted)",
            fontSize: 12,
          }}
        >
          No sessions yet.
        </div>
      )}
      {groups.map((g) => (
        <div key={g.label}>
          <div className="sessions-rail-section">{g.label}</div>
          {g.items.map((s) => {
            const active = s.id === activeId;
            const label = agentNameById[s.agent_id] || s.agent_id;
            const showStatusDot = s.status !== "ready";
            return (
              <button
                key={s.id}
                type="button"
                onClick={() => onPick(s.id)}
                className="sessions-rail-row"
                data-active={active}
              >
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 6,
                    minWidth: 0,
                  }}
                >
                  {showStatusDot && (
                    <span
                      className="sessions-status-dot"
                      style={{ background: statusColor(s.status), margin: 0 }}
                    />
                  )}
                  <span
                    style={{
                      whiteSpace: "nowrap",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      flex: 1,
                      minWidth: 0,
                    }}
                  >
                    {label}
                  </span>
                  <span
                    className="sessions-mono"
                    style={{
                      fontSize: 10,
                      color: "var(--text-muted)",
                      flexShrink: 0,
                    }}
                  >
                    {formatRelative(s.created_at)}
                  </span>
                </div>
              </button>
            );
          })}
        </div>
      ))}
    </aside>
  );
}

function MessageRowView({ msg }: { msg: MessageRow }) {
  if (msg.role === "user") return <UserMessage msg={msg} />;
  return <AssistantMessage msg={msg} />;
}

function UserMessage({ msg }: { msg: MessageRow }) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "flex-start",
        gap: 12,
        margin: "28px 0 16px",
        paddingTop: 16,
        borderTop: "1px solid var(--border-color)",
      }}
    >
      <div
        style={{
          flex: 1,
          fontSize: 13.5,
          lineHeight: 1.55,
          color: "var(--text-primary)",
          fontWeight: 500,
          whiteSpace: "pre-wrap",
          wordBreak: "break-word",
          letterSpacing: "-0.005em",
        }}
      >
        {msg.content}
      </div>
      <span
        className="sessions-mono"
        style={{
          fontSize: 10.5,
          color: "var(--text-muted)",
          flexShrink: 0,
          marginTop: 3,
        }}
      >
        {formatRelative(msg.created_at)}
      </span>
    </div>
  );
}

function AssistantMessage({ msg }: { msg: MessageRow }) {
  const failed = msg.status === "failed";
  const inProgress = msg.status === "in_progress";

  return (
    <div style={{ margin: "8px 0 28px" }}>
      {msg.content ? (
        <div
          className="sessions-md"
          style={{ color: failed ? "#b91c1c" : "var(--text-primary)" }}
        >
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {msg.content}
          </ReactMarkdown>
        </div>
      ) : inProgress ? (
        <div style={{ color: "var(--text-muted)", fontSize: 13.5 }}>
          thinking…
        </div>
      ) : null}
      {failed && msg.error_reason && (
        <div
          className="sessions-mono"
          style={{ fontSize: 11, color: "#b91c1c", marginTop: 6 }}
        >
          {msg.error_reason}
        </div>
      )}
      {msg.tools && msg.tools.length > 0 && (
        <div style={{ marginTop: 12 }}>
          {msg.tools.map((t, i) => (
            <ToolCallView key={i} tool={t} />
          ))}
        </div>
      )}
    </div>
  );
}

function ToolCallView({ tool }: { tool: ToolCall }) {
  const inputStr =
    tool.input === undefined || tool.input === null
      ? ""
      : typeof tool.input === "string"
        ? tool.input
        : JSON.stringify(tool.input, null, 2);
  return (
    <div
      style={{
        border: "1px solid var(--border-color)",
        borderRadius: 6,
        marginBottom: 10,
        background: "var(--bg-code)",
        overflow: "hidden",
      }}
    >
      <div
        className="sessions-mono"
        style={{
          fontSize: 10,
          letterSpacing: 0.5,
          textTransform: "uppercase",
          color: "var(--text-secondary)",
          padding: "6px 10px",
          borderBottom: "1px solid var(--border-color)",
          background: "var(--bg-rail)",
        }}
      >
        tool · {tool.name}
      </div>
      {inputStr && (
        <pre
          className="sessions-mono"
          style={{
            margin: 0,
            padding: "8px 10px",
            fontSize: 11.5,
            color: "var(--text-secondary)",
            whiteSpace: "pre-wrap",
            wordBreak: "break-word",
            maxHeight: 220,
            overflow: "auto",
            borderBottom: tool.output
              ? "1px solid var(--border-color)"
              : "none",
            background: "var(--bg-code)",
          }}
        >
          {inputStr}
        </pre>
      )}
      {tool.output && (
        <pre
          className="sessions-mono"
          style={{
            margin: 0,
            padding: "8px 10px",
            fontSize: 11.5,
            color: "var(--text-primary)",
            whiteSpace: "pre-wrap",
            wordBreak: "break-word",
            maxHeight: 220,
            overflow: "auto",
            background: "var(--bg-code)",
          }}
        >
          {tool.output}
        </pre>
      )}
    </div>
  );
}
