/**
 * @litellm/agent-sdk — public surface.
 *
 * Three-level hierarchy:
 *   Agent       — definition (model, system prompt, tools)
 *     ↓
 *   Session     — VM sandbox running under an agent
 *     ↓
 *   Run         — single execution within a session
 */

export { Agent, AgentHandle } from "./agent.js";
export { SessionHandle } from "./session.js";
export { Run } from "./run.js";

export type {
  AgentCreateOptions,
  AgentInfo,
  ClientOptions,
  ConversationTurn,
  CreateSessionOptions,
  ListOptions,
  ListResult,
  Message,
  ModelRef,
  RunEvent,
  RunInfo,
  RunResult,
  RunStatus,
  SDKImage,
  SessionInfo,
  SessionStatus,
} from "./types.js";

export { LiteLLMAgentError } from "./types.js";
