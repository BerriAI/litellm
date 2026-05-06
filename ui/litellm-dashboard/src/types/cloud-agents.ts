/**
 * Type definitions for the cloud-agents (Cursor SDK on LiteLLM) dashboard.
 *
 * These mirror the API spec from LIT-2877 (Epic A). When the backend lands,
 * this file is the single source of truth — components should import from
 * here, not redefine shapes inline.
 *
 * Namespaced as `Cloud*` to avoid colliding with the existing legacy
 * proxy-side `Agent` type at `src/components/agents/types.ts`.
 */

export type CloudAgentSessionStatus = "provisioning" | "running" | "paused" | "completed" | "failed";

export interface CloudAgent {
  agent_id: string;
  name: string;
  model: string;
  system_prompt: string;
  session_count: number;
  last_activity_at: string | null;
  created_at: string;
}

export interface CloudAgentSession {
  session_id: string;
  agent_id: string;
  repo_url: string;
  branch: string;
  status: CloudAgentSessionStatus;
  title: string;
  active_run_id: string | null;
  created_at: string;
  updated_at: string;
}

export interface CloudAgentRunGitBranch {
  name: string;
  base: string;
}

export interface CloudAgentRunGit {
  branches: CloudAgentRunGitBranch[];
  pr_url: string | null;
}

export interface CloudAgentRun {
  run_id: string;
  session_id: string;
  status: CloudAgentSessionStatus;
  created_at: string;
  terminated_at: string | null;
  git: CloudAgentRunGit;
}

export type CloudAgentConversationRole = "user" | "assistant" | "tool" | "system";

export interface CloudAgentConversationMessage {
  id: string;
  role: CloudAgentConversationRole;
  content: string;
  tool_name?: string;
  created_at: string;
}

/**
 * Event types emitted on the SSE stream from the proxy. The shape of `payload`
 * varies by `type` — keep the discriminated union loose here so we can refine
 * per-component without churning every callsite.
 */
export type CloudAgentRunEventType =
  | "user_message"
  | "assistant_message"
  | "tool_call"
  | "tool_result"
  | "terminal_chunk"
  | "file_diff"
  | "git_commit"
  | "pr_opened"
  | "run_started"
  | "run_completed";

export interface CloudAgentRunEvent {
  seq: number;
  type: CloudAgentRunEventType;
  payload: Record<string, unknown>;
  created_at: string;
}

export interface FileDiffPayload {
  path: string;
  additions: number;
  deletions: number;
  patch: string;
}

export interface TerminalChunkPayload {
  text: string;
}

export interface GitCommitPayload {
  sha: string;
  message: string;
  branch: string;
}

export interface ToolCallPayload {
  tool: string;
  input: string;
}
