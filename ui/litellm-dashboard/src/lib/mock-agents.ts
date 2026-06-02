/**
 * Mock data provider for the cloud-agents dashboard (Cursor SDK on LiteLLM).
 *
 * Wired via env var `NEXT_PUBLIC_USE_MOCK_AGENTS=true`. This is a temporary shim
 * until Epic A (LIT-2877) lands the real `/v2/agents`, `/v2/sessions`,
 * `/v2/sessions/{sid}/conversation` and `/v2/sessions/{sid}/runs/{rid}/events`
 * endpoints. (`/v1/agents` is reserved for the existing A2A registry — the new
 * VM-agent API moves under `/v2/`.) Shapes here mirror the API spec in LIT-2877
 * so the swap is a one-line client change inside `src/lib/cloud-agents-client.ts`.
 *
 * Do not call external services from the UI. Mock data is generated in-process.
 */
import type {
  CloudAgent,
  CloudAgentSession,
  CloudAgentRun,
  CloudAgentRunEvent,
  CloudAgentConversationMessage,
  CloudAgentSessionStatus,
} from "@/types/cloud-agents";

const NOW = () => new Date().toISOString();

const MOCK_AGENTS: CloudAgent[] = [
  {
    agent_id: "agt_01",
    name: "shin-cursor-default",
    model: "claude-3-5-sonnet-20241022",
    system_prompt: "You are a helpful coding agent.",
    session_count: 4,
    last_activity_at: NOW(),
    created_at: "2026-04-01T12:00:00.000Z",
  },
  {
    agent_id: "agt_02",
    name: "code-review-bot",
    model: "claude-3-5-haiku-20241022",
    system_prompt: "You review pull requests.",
    session_count: 1,
    last_activity_at: NOW(),
    created_at: "2026-04-15T12:00:00.000Z",
  },
];

const MOCK_SESSIONS: Record<string, CloudAgentSession[]> = {
  agt_01: [
    {
      session_id: "ses_01",
      agent_id: "agt_01",
      repo_url: "https://github.com/example/repo",
      branch: "feature/refactor-router",
      status: "running" as CloudAgentSessionStatus,
      title: "Refactor router fallback logic",
      active_run_id: "run_01",
      created_at: NOW(),
      updated_at: NOW(),
    },
    {
      session_id: "ses_02",
      agent_id: "agt_01",
      repo_url: "https://github.com/example/repo",
      branch: "fix/sse-reconnect",
      status: "completed" as CloudAgentSessionStatus,
      title: "Fix SSE reconnect bug",
      active_run_id: "run_02",
      created_at: NOW(),
      updated_at: NOW(),
    },
  ],
  agt_02: [
    {
      session_id: "ses_03",
      agent_id: "agt_02",
      repo_url: "https://github.com/example/repo",
      branch: "main",
      status: "provisioning" as CloudAgentSessionStatus,
      title: "Review PR #1234",
      active_run_id: "run_03",
      created_at: NOW(),
      updated_at: NOW(),
    },
  ],
};

const MOCK_CONVERSATION: Record<string, CloudAgentConversationMessage[]> = {
  ses_01: [
    {
      id: "msg_01",
      role: "user",
      content: "Can you refactor the router fallback logic to use the new tags API?",
      created_at: "2026-05-06T20:00:00.000Z",
    },
    {
      id: "msg_02",
      role: "assistant",
      content: "I'll start by reading `litellm/router.py` to understand the current logic.",
      created_at: "2026-05-06T20:00:05.000Z",
    },
    {
      id: "msg_03",
      role: "tool",
      content: "Read litellm/router.py (lines 1-200)",
      tool_name: "read_file",
      created_at: "2026-05-06T20:00:08.000Z",
    },
  ],
  ses_02: [
    {
      id: "msg_10",
      role: "user",
      content: "The SSE reconnect drops events when the connection flaps.",
      created_at: "2026-05-05T10:00:00.000Z",
    },
    {
      id: "msg_11",
      role: "assistant",
      content: "Fixed by tracking the last seq cursor and replaying from there.",
      created_at: "2026-05-05T10:05:00.000Z",
    },
  ],
  ses_03: [],
};

/**
 * Canned event sequence for SSE simulation. Includes assistant streaming,
 * tool calls, file diffs, terminal chunks (one with ANSI red), and a git commit.
 * The mock client replays this with small delays.
 */
export const MOCK_RUN_EVENTS: CloudAgentRunEvent[] = [
  {
    seq: 1,
    type: "user_message",
    payload: { content: "Implement the new feature" },
    created_at: "2026-05-06T20:00:00.000Z",
  },
  {
    seq: 2,
    type: "assistant_message",
    payload: { content: "Starting work on the feature." },
    created_at: "2026-05-06T20:00:01.000Z",
  },
  {
    seq: 3,
    type: "tool_call",
    payload: { tool: "shell", input: "pnpm install" },
    created_at: "2026-05-06T20:00:02.000Z",
  },
  {
    seq: 4,
    type: "terminal_chunk",
    payload: { text: "[31mERROR[0m: missing dep\n" },
    created_at: "2026-05-06T20:00:03.000Z",
  },
  {
    seq: 5,
    type: "file_diff",
    payload: {
      path: "src/index.ts",
      additions: 12,
      deletions: 3,
      patch: "@@ -1,3 +1,12 @@\n+import { foo } from './foo';",
    },
    created_at: "2026-05-06T20:00:04.000Z",
  },
  {
    seq: 6,
    type: "git_commit",
    payload: { sha: "abc123", message: "feat: add new feature", branch: "feature/new" },
    created_at: "2026-05-06T20:00:05.000Z",
  },
];

const MOCK_RUNS: Record<string, CloudAgentRun> = {
  run_01: {
    run_id: "run_01",
    session_id: "ses_01",
    status: "running",
    created_at: "2026-05-06T20:00:00.000Z",
    terminated_at: null,
    git: { branches: [{ name: "feature/refactor-router", base: "main" }], pr_url: null },
  },
  run_02: {
    run_id: "run_02",
    session_id: "ses_02",
    status: "completed",
    created_at: "2026-05-05T10:00:00.000Z",
    terminated_at: "2026-05-05T10:05:00.000Z",
    git: {
      branches: [{ name: "fix/sse-reconnect", base: "main" }],
      pr_url: "https://github.com/example/repo/pull/42",
    },
  },
  run_03: {
    run_id: "run_03",
    session_id: "ses_03",
    status: "provisioning",
    created_at: NOW(),
    terminated_at: null,
    git: { branches: [], pr_url: null },
  },
};

export function isMockEnabled(): boolean {
  return process.env.NEXT_PUBLIC_USE_MOCK_AGENTS === "true";
}

/* ---------- mock client surface ---------- */

export async function mockListAgents(): Promise<CloudAgent[]> {
  return MOCK_AGENTS;
}

export async function mockGetAgent(agentId: string): Promise<CloudAgent | null> {
  return MOCK_AGENTS.find((a) => a.agent_id === agentId) ?? null;
}

export async function mockListSessions(agentId: string): Promise<CloudAgentSession[]> {
  return MOCK_SESSIONS[agentId] ?? [];
}

export async function mockGetSession(sessionId: string): Promise<CloudAgentSession | null> {
  for (const sessions of Object.values(MOCK_SESSIONS)) {
    const found = sessions.find((s) => s.session_id === sessionId);
    if (found) return found;
  }
  return null;
}

export async function mockGetRun(runId: string): Promise<CloudAgentRun | null> {
  return MOCK_RUNS[runId] ?? null;
}

export async function mockGetConversation(sessionId: string): Promise<CloudAgentConversationMessage[]> {
  return MOCK_CONVERSATION[sessionId] ?? [];
}

export async function mockCreateAgent(input: {
  name: string;
  model: string;
  system_prompt?: string;
}): Promise<CloudAgent> {
  const agent: CloudAgent = {
    agent_id: `agt_${Math.random().toString(36).slice(2, 8)}`,
    name: input.name,
    model: input.model,
    system_prompt: input.system_prompt ?? "",
    session_count: 0,
    last_activity_at: NOW(),
    created_at: NOW(),
  };
  MOCK_AGENTS.push(agent);
  return agent;
}

export async function mockCreateSession(input: { agent_id: string; repo_url: string }): Promise<CloudAgentSession> {
  const session: CloudAgentSession = {
    session_id: `ses_${Math.random().toString(36).slice(2, 8)}`,
    agent_id: input.agent_id,
    repo_url: input.repo_url,
    branch: "main",
    status: "provisioning",
    title: "New session",
    active_run_id: null,
    created_at: NOW(),
    updated_at: NOW(),
  };
  if (!MOCK_SESSIONS[input.agent_id]) {
    MOCK_SESSIONS[input.agent_id] = [];
  }
  MOCK_SESSIONS[input.agent_id].push(session);
  return session;
}

export async function mockSendFollowup(sessionId: string, content: string): Promise<CloudAgentConversationMessage> {
  const msg: CloudAgentConversationMessage = {
    id: `msg_${Math.random().toString(36).slice(2, 8)}`,
    role: "user",
    content,
    created_at: NOW(),
  };
  if (!MOCK_CONVERSATION[sessionId]) {
    MOCK_CONVERSATION[sessionId] = [];
  }
  MOCK_CONVERSATION[sessionId].push(msg);
  return msg;
}
