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
import {
  PanelLeft,
  Search,
  Circle,
  ChevronDown,
  ChevronRight,
  CheckCircle2,
  Folder,
  MoreHorizontal,
  PanelRight,
  ArrowUp,
  Square,
  Image as ImageIcon,
  Loader2,
} from "lucide-react";

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

interface AgentRow {
  id: string;
  name: string;
}

const DEFAULT_PROXY = "http://localhost:4000";
const DEFAULT_KEY = "sk-1234";
const POLL_INTERVAL_MS = 2000;

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
    const diff = Math.max(0, Date.now() - d);
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
    return `${Math.floor(day / 30)}mo`;
  } catch {
    return "";
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
    "This Month": [],
    Older: [],
  };
  for (const s of sessions) {
    const age = now - new Date(s.created_at).getTime();
    if (age < day) buckets.Today.push(s);
    else if (age < 7 * day) buckets["This Week"].push(s);
    else if (age < 30 * day) buckets["This Month"].push(s);
    else buckets.Older.push(s);
  }
  return [
    { label: "Today", items: buckets.Today },
    { label: "This Week", items: buckets["This Week"] },
    { label: "This Month", items: buckets["This Month"] },
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
  const [pendingMessageId, setPendingMessageId] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);

  // Only ASSISTANT messages count as "in flight". User messages from opencode
  // have no completed_at so they normalize as in_progress, but they're not
  // running anything — only assistant turns are.
  const hasInProgress = useMemo(
    () =>
      messages.some(
        (m) => m.role === "assistant" && m.status === "in_progress",
      ),
    [messages],
  );

  const currentModel = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      const m = messages[i];
      if (m.role === "assistant" && m.model) return m.model;
    }
    return session?.default_model || "";
  }, [messages, session]);

  const currentAgentName = useMemo(() => {
    if (session?.agent_name) return session.agent_name;
    if (session) return agentNameById[session.agent_id] || session.agent_id;
    return "";
  }, [session, agentNameById]);

  // Load this session + messages
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

  // Load all sessions + agents for the rail; poll every 5s so externally-created
  // sessions (e.g. curl POST /v2/sessions) appear without reload.
  useEffect(() => {
    let cancelled = false;
    const fetchAll = async () => {
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
          const data: ListResponse<AgentRow> = await aRes.json();
          const map: Record<string, string> = {};
          for (const a of data.data || []) map[a.id] = a.name;
          setAgentNameById(map);
        }
      } catch {
        // silent
      }
    };
    fetchAll();
    const interval = setInterval(fetchAll, 5000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [sessionId]);

  // Always poll messages (so externally-sent messages appear without reload).
  // Faster when something is in_progress; slower when idle.
  useEffect(() => {
    if (!sessionId) return;
    let cancelled = false;
    const fetchMessages = async () => {
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
    };
    const intervalMs = hasInProgress ? POLL_INTERVAL_MS : 5000;
    const interval = setInterval(fetchMessages, intervalMs);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [sessionId, hasInProgress]);

  // Auto-scroll only when user is already near the bottom; don't hijack scroll
  const lastMessageCountRef = useRef<number>(0);
  useEffect(() => {
    const c = scrollContainerRef.current;
    if (!c) return;
    const newCount = messages.length;
    const grew = newCount > lastMessageCountRef.current;
    lastMessageCountRef.current = newCount;
    const distanceFromBottom = c.scrollHeight - c.scrollTop - c.clientHeight;
    const nearBottom = distanceFromBottom < 200;
    if (grew && nearBottom) {
      messagesEndRef.current?.scrollIntoView({
        behavior: "smooth",
        block: "end",
      });
    }
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
      status: "in_progress", // queued — server hasn't confirmed yet
      created_at: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, optimistic]);
    setPendingMessageId(optimisticId);
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
      setPendingMessageId(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setMessages((prev) => prev.filter((x) => x.id !== optimisticId));
      setPendingMessageId(null);
    } finally {
      setSending(false);
    }
  }, [draft, sessionId, sending]);

  const handleAbort = useCallback(async () => {
    if (!sessionId || aborting) return;
    setAborting(true);
    try {
      await fetch(`${getProxyBase()}/v2/sessions/${sessionId}/abort`, {
        method: "POST",
        headers: buildHeaders(),
      });
      const refreshed = await fetch(
        `${getProxyBase()}/v2/sessions/${sessionId}/messages`,
        { headers: buildHeaders() },
      );
      if (refreshed.ok) {
        const m: ListResponse<MessageRow> = await refreshed.json();
        setMessages(m.data || []);
      }
    } catch {
      // silent
    } finally {
      setAborting(false);
    }
  }, [sessionId, aborting]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      // Enter sends (no modifiers). Shift+Enter inserts a newline.
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend],
  );

  const isQueued = pendingMessageId !== null || sending;

  return (
    <div className="sessions-app flex w-full h-screen bg-white text-gray-900 overflow-hidden">
      <Sidebar
        sessions={sessionsList}
        agentNameById={agentNameById}
        activeSessionId={sessionId}
        onPick={(id) => router.push(`/sessions/${id}`)}
      />
      <MainPanel
        session={session}
        agentName={currentAgentName}
        messages={messages}
        loading={loading}
        error={error}
        sending={sending}
        aborting={aborting}
        hasInProgress={hasInProgress || isQueued}
        currentModel={currentModel}
        draft={draft}
        setDraft={setDraft}
        handleSend={handleSend}
        handleAbort={handleAbort}
        handleKeyDown={handleKeyDown}
        messagesEndRef={messagesEndRef}
        scrollContainerRef={scrollContainerRef}
        pendingMessageId={pendingMessageId}
      />
    </div>
  );
}

// =====================================================================
// SIDEBAR
// =====================================================================

function Sidebar({
  sessions,
  agentNameById,
  activeSessionId,
  onPick,
}: {
  sessions: SessionRow[];
  agentNameById: Record<string, string>;
  activeSessionId: string;
  onPick: (id: string) => void;
}) {
  const groups = useMemo(() => groupSessionsByAge(sessions), [sessions]);

  return (
    <div className="w-[260px] flex-shrink-0 bg-[#fbfbfb] border-r border-gray-200 flex flex-col h-screen text-sm select-none">
      {/* Top header: hamburger + search */}
      <div className="flex items-center gap-2 p-3">
        <button className="p-1 hover:bg-gray-200 rounded text-gray-500">
          <PanelLeft className="w-4 h-4" />
        </button>
        <div className="flex-1 flex items-center gap-2 bg-white border border-gray-200 rounded-md px-2 py-1.5 text-gray-400 shadow-sm">
          <Search className="w-3.5 h-3.5" />
          <span className="text-xs">Search sessions ⌘K</span>
        </div>
      </div>

      {/* Session groups */}
      <div className="flex-1 overflow-y-auto mt-2">
        {groups.length === 0 && (
          <div className="px-4 text-[12px] text-gray-400">No sessions yet.</div>
        )}
        {groups.map((g) => (
          <React.Fragment key={g.label}>
            <div className="px-4 mt-2 mb-1">
              <span className="text-[11px] font-medium text-gray-400 uppercase tracking-wider">
                {g.label}
              </span>
            </div>
            <div className="px-2 space-y-0.5">
              {g.items.map((s) => {
                const active = s.id === activeSessionId;
                const label = agentNameById[s.agent_id] || s.agent_id;
                return active ? (
                  <ActiveItem key={s.id} label={label} status={s.status} />
                ) : (
                  <HistoryItem
                    key={s.id}
                    label={label}
                    onClick={() => onPick(s.id)}
                  />
                );
              })}
            </div>
          </React.Fragment>
        ))}
      </div>

      {/* User footer */}
      <div className="p-3 border-t border-gray-200 flex items-center gap-2.5">
        <div className="w-7 h-7 rounded-full bg-indigo-600 text-white flex items-center justify-center text-xs font-medium">
          I
        </div>
        <div className="flex flex-col">
          <span className="text-[13px] font-medium text-gray-700 leading-tight">
            Ishaan Jaffer
          </span>
          <span className="text-[11px] text-gray-500 leading-tight">
            LiteLLM
          </span>
        </div>
      </div>
    </div>
  );
}

function ActiveItem({ label, status }: { label: string; status: string }) {
  const dotColor =
    status === "ready"
      ? "fill-emerald-500 text-emerald-500"
      : status === "error" || status === "failed"
        ? "fill-red-500 text-red-500"
        : "fill-blue-500 text-blue-500";
  return (
    <div className="flex items-center justify-between bg-[#eef2ff] text-gray-900 px-2 py-1.5 rounded-md cursor-pointer">
      <div className="flex items-center gap-2 overflow-hidden">
        <Circle className={`w-2 h-2 ${dotColor} flex-shrink-0`} />
        <span className="truncate text-[13px] font-medium">{label}</span>
      </div>
    </div>
  );
}

function HistoryItem({
  label,
  onClick,
}: {
  label: string;
  onClick: () => void;
}) {
  return (
    <div
      onClick={onClick}
      className="flex items-center justify-between px-2 py-1.5 text-gray-600 hover:bg-gray-100 rounded-md cursor-pointer group"
    >
      <div className="flex items-center gap-2 overflow-hidden">
        <span className="truncate text-[13px]">{label}</span>
      </div>
    </div>
  );
}

// =====================================================================
// MAIN PANEL
// =====================================================================

interface MainPanelProps {
  session: SessionRow | null;
  agentName: string;
  messages: MessageRow[];
  loading: boolean;
  error: string | null;
  sending: boolean;
  aborting: boolean;
  hasInProgress: boolean;
  currentModel: string;
  draft: string;
  setDraft: (s: string) => void;
  handleSend: () => void;
  handleAbort: () => void;
  handleKeyDown: (e: React.KeyboardEvent<HTMLTextAreaElement>) => void;
  messagesEndRef: React.RefObject<HTMLDivElement>;
  scrollContainerRef: React.RefObject<HTMLDivElement>;
  pendingMessageId: string | null;
}

function MainPanel({
  session,
  agentName,
  messages,
  loading,
  error,
  sending,
  aborting,
  hasInProgress,
  currentModel,
  draft,
  setDraft,
  handleSend,
  handleAbort,
  handleKeyDown,
  messagesEndRef,
  scrollContainerRef,
  pendingMessageId,
}: MainPanelProps) {
  const repoLabel = session?.repos?.[0]?.url
    ? session.repos[0].url.replace(/^https?:\/\/github\.com\//, "")
    : "BerriAI/litellm";

  return (
    <div className="flex-1 flex flex-col h-screen min-h-0 bg-white overflow-hidden">
      {/* Header */}
      <div className="h-12 border-b border-gray-200 flex items-center justify-between px-4 flex-shrink-0">
        <div className="flex items-center gap-2 text-[13px] text-gray-600">
          <span className="font-medium text-gray-800">
            {agentName || "Session"}
          </span>
          <span className="text-gray-300">/</span>
          <div className="flex items-center gap-1.5 hover:bg-gray-100 px-1.5 py-1 rounded cursor-pointer">
            <Folder className="w-3.5 h-3.5 text-gray-400" />
            <span>{repoLabel}</span>
          </div>
        </div>
        <div className="flex items-center gap-2 text-gray-400">
          <button className="p-1.5 hover:bg-gray-100 rounded">
            <MoreHorizontal className="w-4 h-4" />
          </button>
          <button className="p-1.5 hover:bg-gray-100 rounded">
            <PanelRight className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Scrollable thread (composer is OUTSIDE this scroll) */}
      <div
        ref={scrollContainerRef}
        className="flex-1 min-h-0 overflow-y-auto"
      >
        <div className="max-w-[720px] mx-auto w-full py-10 px-6 flex flex-col gap-6">
          {loading && messages.length === 0 && (
            <div className="text-[13px] text-gray-400">Loading…</div>
          )}
          {!loading && messages.length === 0 && (
            <div className="text-[13px] text-gray-400">
              No messages. Send one below.
            </div>
          )}

          {messages.map((m, i) => (
            <MessageBlock
              key={m.id}
              msg={m}
              isFirstUser={
                m.role === "user" &&
                messages.slice(0, i).every((x) => x.role !== "user")
              }
              isPending={m.id === pendingMessageId}
            />
          ))}

          <div ref={messagesEndRef} />
          <div className="h-4" />
        </div>
      </div>

      {/* Sticky composer */}
      <div className="flex-shrink-0 border-t border-gray-200 bg-white">
        <div className="max-w-[720px] mx-auto w-full px-6 py-4">
          <Composer
            draft={draft}
            setDraft={setDraft}
            sending={sending}
            aborting={aborting}
            hasInProgress={hasInProgress}
            currentModel={currentModel}
            error={error}
            handleSend={handleSend}
            handleAbort={handleAbort}
            handleKeyDown={handleKeyDown}
          />
        </div>
      </div>
    </div>
  );
}

function MessageBlock({
  msg,
  isFirstUser,
  isPending,
}: {
  msg: MessageRow;
  isFirstUser: boolean;
  isPending: boolean;
}) {
  if (msg.role === "user") {
    return (
      <UserPromptBlock
        content={msg.content}
        emphasized={isFirstUser}
        pending={isPending}
      />
    );
  }
  return <AssistantBlock msg={msg} />;
}

function UserPromptBlock({
  content,
  emphasized,
  pending,
}: {
  content: string;
  emphasized: boolean;
  pending: boolean;
}) {
  return (
    <div
      className={`bg-[#f9f9f9] border border-gray-100 rounded-xl p-4 text-[14px] text-gray-700 leading-relaxed whitespace-pre-wrap ${
        emphasized ? "shadow-sm" : ""
      }`}
    >
      {content}
      {pending && (
        <div className="mt-3 flex items-center gap-1.5 text-[11px] text-gray-400 font-mono">
          <Loader2 className="w-3 h-3 animate-spin" />
          <span>queued</span>
        </div>
      )}
    </div>
  );
}

function AssistantBlock({ msg }: { msg: MessageRow }) {
  const failed = msg.status === "failed";
  const inProgress = msg.status === "in_progress";

  return (
    <div className="flex flex-col gap-3">
      {msg.content ? (
        <div
          className="sessions-md text-[14px] text-gray-800 leading-relaxed"
          style={{ color: failed ? "#b91c1c" : undefined }}
        >
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {msg.content}
          </ReactMarkdown>
        </div>
      ) : inProgress ? (
        <div className="text-[14px] text-gray-400 leading-relaxed">
          thinking…
        </div>
      ) : null}

      {failed && msg.error_reason && (
        <div className="font-mono text-[11px] text-red-700">
          {msg.error_reason}
        </div>
      )}

      {msg.tools && msg.tools.length > 0 && (
        <div className="flex flex-col gap-2">
          {msg.tools.map((t, i) => (
            <ToolResultCard key={i} tool={t} />
          ))}
        </div>
      )}
    </div>
  );
}

function ToolResultCard({ tool }: { tool: ToolCall }) {
  const [expanded, setExpanded] = useState(false);
  const inputStr =
    tool.input === undefined || tool.input === null
      ? ""
      : typeof tool.input === "string"
        ? tool.input
        : JSON.stringify(tool.input, null, 2);
  const succeeded = !!tool.output;

  return (
    <div>
      <div
        onClick={() => setExpanded((e) => !e)}
        className="border border-gray-200 rounded-xl p-4 flex items-center gap-4 bg-[#fcfcfc] shadow-sm cursor-pointer hover:bg-gray-50 transition-colors"
      >
        <div
          className={`w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0 ${
            succeeded
              ? "bg-emerald-50 border border-emerald-100"
              : "bg-amber-50 border border-amber-100"
          }`}
        >
          <CheckCircle2
            className={`w-5 h-5 ${succeeded ? "text-emerald-500" : "text-amber-500"}`}
          />
        </div>
        <div className="flex flex-col flex-1 min-w-0">
          <span className="text-[14px] font-medium text-gray-800">
            {tool.name}
          </span>
          <span className="text-[12px] text-gray-500 truncate">
            {succeeded ? "Completed" : "Pending"}
          </span>
        </div>
        {expanded ? (
          <ChevronDown className="w-4 h-4 text-gray-400 flex-shrink-0" />
        ) : (
          <ChevronRight className="w-4 h-4 text-gray-400 flex-shrink-0" />
        )}
      </div>
      {expanded && (
        <div className="mt-2 border border-gray-200 rounded-lg bg-[#fcfcfc] overflow-hidden">
          {inputStr && (
            <pre className="m-0 p-3 font-mono text-[12px] text-gray-600 whitespace-pre-wrap break-words border-b border-gray-200 max-h-60 overflow-auto">
              {inputStr}
            </pre>
          )}
          {tool.output && (
            <pre className="m-0 p-3 font-mono text-[12px] text-gray-800 whitespace-pre-wrap break-words max-h-60 overflow-auto">
              {tool.output}
            </pre>
          )}
        </div>
      )}
    </div>
  );
}

// =====================================================================
// COMPOSER
// =====================================================================

interface ComposerProps {
  draft: string;
  setDraft: (s: string) => void;
  sending: boolean;
  aborting: boolean;
  hasInProgress: boolean;
  currentModel: string;
  error: string | null;
  handleSend: () => void;
  handleAbort: () => void;
  handleKeyDown: (e: React.KeyboardEvent<HTMLTextAreaElement>) => void;
}

function Composer({
  draft,
  setDraft,
  sending,
  aborting,
  hasInProgress,
  currentModel,
  error,
  handleSend,
  handleAbort,
  handleKeyDown,
}: ComposerProps) {
  const canSend = draft.trim().length > 0 && !sending;

  return (
    <div className="border border-gray-200 rounded-xl shadow-sm bg-white overflow-hidden focus-within:ring-1 focus-within:ring-gray-300 focus-within:border-gray-300 transition-all">
      <textarea
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Add a follow up"
        disabled={sending}
        rows={1}
        className="w-full p-4 outline-none resize-none text-[15px] placeholder:text-gray-400 bg-transparent"
      />
      <div className="flex items-center justify-between px-4 pb-3 text-xs text-gray-500">
        <span className="font-mono">
          {error ? (
            <span className="text-red-600">{error}</span>
          ) : (
            currentModel || "Enter to send · Shift+Enter for newline"
          )}
        </span>
        <div className="flex items-center gap-3">
          <button
            type="button"
            className="hover:text-gray-700 transition-colors"
            aria-label="Attach"
          >
            <ImageIcon className="w-4 h-4" />
          </button>
          {hasInProgress ? (
            <button
              type="button"
              onClick={handleAbort}
              disabled={aborting}
              className="bg-black text-white p-1.5 rounded-full hover:bg-gray-800 transition-colors disabled:opacity-50"
              aria-label="Stop"
              title="Stop"
            >
              <Square className="w-3 h-3 fill-current" />
            </button>
          ) : (
            <button
              type="button"
              onClick={handleSend}
              disabled={!canSend}
              className="bg-black text-white p-1.5 rounded-full hover:bg-gray-800 transition-colors disabled:opacity-30 disabled:hover:bg-black"
              aria-label="Send"
              title="Send (Enter)"
            >
              <ArrowUp className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
