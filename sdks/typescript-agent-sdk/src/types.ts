/**
 * Public types for @litellm/agent-sdk.
 *
 * These mirror the wire shapes posted by Epic A on LIT-2880. When A1
 * publishes the canonical OpenAPI/wire-shape comment, regenerate the
 * `Wire*` aliases below; the public surface (Agent/Session/Run) should
 * remain stable.
 */

/** Configuration shared by every SDK call. */
export interface ClientOptions {
  /** LiteLLM API key. Falls back to `process.env.LITELLM_API_KEY`. */
  apiKey?: string;
  /** Base URL of the LiteLLM proxy. Defaults to `https://api.litellm.ai`. */
  baseUrl?: string;
  /** Optional fetch override for tests / custom transports. */
  fetch?: typeof fetch;
  /** Default request timeout (ms). Default: 60_000. */
  timeoutMs?: number;
  /** Max retry attempts on retryable errors. Default: 3. */
  maxRetries?: number;
}

/** Pagination args accepted by `list*` methods. */
export interface ListOptions {
  limit?: number;
  cursor?: string;
}

export interface ListResult<T> {
  items: T[];
  nextCursor?: string;
}

/** Used to identify the model an agent should run on. */
export interface ModelRef {
  /** LiteLLM model ID, e.g. `claude-4.6-sonnet`. */
  id: string;
}

/** Image attachment passed to `session.send`. */
export interface SDKImage {
  /** Either a data URL or an https URL. */
  url: string;
  /** Optional MIME type hint. */
  mimeType?: string;
}

/** Args accepted by `Agent.create`. */
export interface AgentCreateOptions extends ClientOptions {
  name: string;
  model: ModelRef;
  systemPrompt?: string;
  /** Free-form tags / metadata persisted on the agent definition. */
  metadata?: Record<string, string>;
}

/** Args accepted by `agent.createSession`. */
export interface CreateSessionOptions {
  /** Repos to clone into the session VM. */
  repos?: { url: string; startingRef?: string }[];
  /** Environment variables surfaced to the VM. */
  envVars?: Record<string, string>;
  /** Free-form tags / metadata persisted on the session. */
  metadata?: Record<string, string>;
}

export type SessionStatus =
  | "provisioning"
  | "ready"
  | "busy"
  | "error"
  | "terminated";

export type RunStatus =
  | "queued"
  | "running"
  | "finished"
  | "cancelled"
  | "error";

/** Lightweight info returned by list endpoints. */
export interface AgentInfo {
  id: string;
  name: string;
  model: ModelRef;
  createdAt: string;
}

export interface SessionInfo {
  id: string;
  agentId: string;
  status: SessionStatus;
  createdAt: string;
}

export interface RunInfo {
  id: string;
  sessionId: string;
  status: RunStatus;
  result: string | null;
  startedAt: string | null;
  completedAt: string | null;
  git?: { branches: { branch: string; prUrl: string | null }[] };
}

export interface ConversationTurn {
  role: "user" | "assistant" | "system" | "tool";
  content: string;
  runId?: string;
  createdAt: string;
}

export interface Message extends ConversationTurn {}

/** Streamed event off `run.stream()`. */
export interface RunEvent {
  /** Monotonic per-run sequence; required for resume. */
  seq: number;
  type:
    | "delta"
    | "tool_call"
    | "tool_result"
    | "status"
    | "done"
    | "error"
    | string;
  data: unknown;
}

/** Resolved value from `run.wait()`. */
export interface RunResult {
  id: string;
  status: RunStatus;
  result: string | null;
  git?: { branches: { branch: string; prUrl: string | null }[] };
}

/** Error thrown by the SDK. Mirrors Cursor SDK's `CursorAgentError` shape. */
export class LiteLLMAgentError extends Error {
  /** Stable machine-readable code, e.g. `not_found`, `rate_limited`. */
  code: string;
  /** HTTP status, when applicable. */
  status?: number;
  /** Whether the call is safe to retry. */
  retryable: boolean;

  constructor(
    message: string,
    options: { code: string; status?: number; retryable?: boolean } = {
      code: "unknown",
    },
  ) {
    super(message);
    this.name = "LiteLLMAgentError";
    this.code = options.code;
    this.status = options.status;
    this.retryable = options.retryable ?? false;
  }
}
