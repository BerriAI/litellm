/**
 * Agent — the *definition* of an agent (model, system prompt, tools).
 *
 * An Agent is not a VM. Use `agent.createSession()` to spin up a VM that runs
 * under this agent definition.
 */

import { requestJson, resolveClient, type ResolvedClient } from "./client/http.js";
import { SessionHandle } from "./session.js";
import {
  type AgentCreateOptions,
  type AgentInfo,
  type ClientOptions,
  type CreateSessionOptions,
  type ListOptions,
  type ListResult,
  type SessionInfo,
} from "./types.js";

export class AgentHandle {
  readonly id: string;
  readonly name: string;

  private readonly _client: ResolvedClient;

  constructor(info: AgentInfo, client: ResolvedClient) {
    this.id = info.id;
    this.name = info.name;
    this._client = client;
  }

  /** Create a new session running under this agent. */
  async createSession(options: CreateSessionOptions = {}): Promise<SessionHandle> {
    const info = await requestJson<SessionInfo>(this._client, {
      method: "POST",
      path: `/v2/agents/${encodeURIComponent(this.id)}/sessions`,
      body: {
        repos: options.repos ?? [],
        envVars: options.envVars ?? {},
        metadata: options.metadata ?? {},
      },
    });
    return new SessionHandle(info, this._client);
  }

  /** Fetch a session by ID under this agent. */
  async getSession(sessionId: string): Promise<SessionHandle> {
    const info = await requestJson<SessionInfo>(this._client, {
      method: "GET",
      path: `/v2/agents/${encodeURIComponent(this.id)}/sessions/${encodeURIComponent(sessionId)}`,
    });
    return new SessionHandle(info, this._client);
  }

  /** List sessions running under this agent. */
  async listSessions(options: ListOptions = {}): Promise<ListResult<SessionInfo>> {
    const data = await requestJson<{ items: SessionInfo[]; nextCursor?: string }>(
      this._client,
      {
        method: "GET",
        path: `/v2/agents/${encodeURIComponent(this.id)}/sessions`,
        query: { limit: options.limit, cursor: options.cursor },
      }
    );
    return { items: data.items ?? [], nextCursor: data.nextCursor };
  }

  /** Patch the agent definition. */
  async update(patch: Partial<AgentCreateOptions>): Promise<void> {
    await requestJson<void>(this._client, {
      method: "PATCH",
      path: `/v2/agents/${encodeURIComponent(this.id)}`,
      body: stripClientOptions(patch),
    });
  }

  /** Delete the agent definition. Cascades to its sessions. */
  async delete(): Promise<void> {
    await requestJson<void>(this._client, {
      method: "DELETE",
      path: `/v2/agents/${encodeURIComponent(this.id)}`,
    });
  }
}

export class Agent {
  /** Create a new agent definition. */
  static async create(options: AgentCreateOptions): Promise<AgentHandle> {
    const client = resolveClient(options);
    const info = await requestJson<AgentInfo>(client, {
      method: "POST",
      path: "/v2/agents",
      body: stripClientOptions(options),
    });
    return new AgentHandle(info, client);
  }

  /** Fetch an existing agent by ID. */
  static async get(agentId: string, options: ClientOptions = {}): Promise<AgentHandle> {
    const client = resolveClient(options);
    const info = await requestJson<AgentInfo>(client, {
      method: "GET",
      path: `/v2/agents/${encodeURIComponent(agentId)}`,
    });
    return new AgentHandle(info, client);
  }

  /** List agents accessible to this API key. */
  static async list(
    options: ClientOptions & ListOptions = {}
  ): Promise<ListResult<AgentInfo>> {
    const client = resolveClient(options);
    const data = await requestJson<{ items: AgentInfo[]; nextCursor?: string }>(
      client,
      {
        method: "GET",
        path: "/v2/agents",
        query: { limit: options.limit, cursor: options.cursor },
      }
    );
    return { items: data.items ?? [], nextCursor: data.nextCursor };
  }
}

function stripClientOptions<T extends Partial<AgentCreateOptions>>(
  obj: T
): Omit<T, "apiKey" | "baseUrl" | "fetch" | "timeoutMs" | "maxRetries"> {
  const { apiKey, baseUrl, fetch, timeoutMs, maxRetries, ...rest } = obj as AgentCreateOptions;
  void apiKey;
  void baseUrl;
  void fetch;
  void timeoutMs;
  void maxRetries;
  return rest as Omit<T, "apiKey" | "baseUrl" | "fetch" | "timeoutMs" | "maxRetries">;
}
