/**
 * XctClient — TypeScript SDK for the xct-litellm capability provider.
 *
 * Mirrors the Python SDK surface. fetch-only (Node 18+ / browser). Stream
 * helpers (S5-04) are first-class — chat.completions.create({stream:true})
 * and agents.invoke({stream:true}) return AsyncIterables.
 */

import {
  CapabilityNotFoundError,
  fromResponse,
  XctError,
} from "./errors";

export interface XctClientConfig {
  baseUrl: string;
  accessToken?: string;
  appId?: string;
  fetch?: typeof fetch;
  timeoutMs?: number;
}

interface RequestArgs {
  params?: Record<string, string | number | boolean | undefined | null>;
  body?: unknown;
  acceptSse?: boolean;
}

export class XctClient {
  readonly baseUrl: string;
  accessToken?: string;
  readonly appId?: string;
  private readonly _fetch: typeof fetch;
  private readonly timeoutMs: number;

  readonly capabilities: CapabilitiesResource;
  readonly agents: AgentsResource;
  readonly mcp: McpResource;
  readonly skills: SkillsResource;
  readonly chat: ChatResource;

  constructor(config: XctClientConfig) {
    this.baseUrl = config.baseUrl.replace(/\/$/, "");
    this.accessToken = config.accessToken;
    this.appId = config.appId;
    this._fetch = config.fetch ?? globalThis.fetch.bind(globalThis);
    this.timeoutMs = config.timeoutMs ?? 60_000;

    this.capabilities = new CapabilitiesResource(this);
    this.agents = new AgentsResource(this);
    this.mcp = new McpResource(this);
    this.skills = new SkillsResource(this);
    this.chat = new ChatResource(this);
  }

  /** Returns the JSON body for 2xx, throws an XctError for 4xx/5xx. */
  async request<T = unknown>(method: string, path: string, args: RequestArgs = {}): Promise<T> {
    const url = this._url(path, args.params);
    const headers = this._headers();
    if (args.body !== undefined) headers["Content-Type"] = "application/json";

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    try {
      const resp = await this._fetch(url, {
        method,
        headers,
        body: args.body !== undefined ? JSON.stringify(args.body) : undefined,
        signal: controller.signal,
      });
      if (resp.status === 204) return undefined as unknown as T;
      const ct = resp.headers.get("content-type") ?? "";
      const text = await resp.text();
      const parsed = ct.includes("application/json") && text ? JSON.parse(text) : text;
      if (resp.ok) return parsed as T;
      throw fromResponse(resp.status, parsed);
    } finally {
      clearTimeout(timer);
    }
  }

  /** Open a streaming response. Yields parsed events from SSE or NDJSON. */
  async *stream<T = unknown>(
    method: string,
    path: string,
    args: RequestArgs = {},
  ): AsyncIterable<T> {
    const url = this._url(path, args.params);
    const headers = this._headers();
    if (args.acceptSse) headers["Accept"] = "text/event-stream";
    if (args.body !== undefined) headers["Content-Type"] = "application/json";

    const resp = await this._fetch(url, {
      method,
      headers,
      body: args.body !== undefined ? JSON.stringify(args.body) : undefined,
    });
    if (!resp.ok || !resp.body) {
      const text = resp.body ? await resp.text() : "";
      throw fromResponse(resp.status, _safeJson(text));
    }
    const isSse = (resp.headers.get("content-type") ?? "").includes("text/event-stream");
    if (isSse) {
      yield* parseSse<T>(resp.body);
    } else {
      yield* parseNdjson<T>(resp.body);
    }
  }

  // ---- internals ------------------------------------------------------

  private _url(path: string, params?: RequestArgs["params"]): string {
    const base = path.startsWith("/") ? this.baseUrl + path : `${this.baseUrl}/${path}`;
    if (!params) return base;
    const usp = new URLSearchParams();
    for (const [k, v] of Object.entries(params)) {
      if (v !== undefined && v !== null) usp.set(k, String(v));
    }
    const qs = usp.toString();
    return qs ? `${base}?${qs}` : base;
  }

  private _headers(): Record<string, string> {
    const h: Record<string, string> = { Accept: "application/json" };
    if (this.accessToken) h["Authorization"] = `Bearer ${this.accessToken}`;
    if (this.appId) h["x-xct-app-id"] = this.appId;
    return h;
  }
}

// ============================================================================
// Resources
// ============================================================================

class _Resource {
  constructor(protected readonly client: XctClient) {}
}

class CapabilitiesResource extends _Resource {
  list(): Promise<unknown> {
    return this.client.request("GET", "/v1/capabilities");
  }
}

class AgentsResource extends _Resource {
  list(params: {
    q?: string;
    category?: string;
    tag?: string;
    cursor?: string;
    limit?: number;
  } = {}): Promise<unknown[]> {
    return this.client.request("GET", "/v1/agents", { params });
  }

  get(agentId: string): Promise<unknown> {
    return this.client.request("GET", `/v1/agents/${encodeURIComponent(agentId)}`);
  }

  /** Invoke an A2A agent. Set stream:true to get an AsyncIterable of events. */
  invoke(
    agentId: string,
    args: { message: Record<string, unknown>; requestId?: string; stream?: false },
  ): Promise<unknown>;
  invoke(
    agentId: string,
    args: { message: Record<string, unknown>; requestId?: string; stream: true },
  ): AsyncIterable<unknown>;
  invoke(
    agentId: string,
    args: { message: Record<string, unknown>; requestId?: string; stream?: boolean },
  ): Promise<unknown> | AsyncIterable<unknown> {
    const body = {
      jsonrpc: "2.0",
      id: args.requestId ?? "1",
      method: args.stream ? "message/stream" : "message/send",
      params: { message: args.message },
    };
    const path = `/v1/a2a/${encodeURIComponent(agentId)}/message/send`;
    if (args.stream) {
      return this.client.stream("POST", path, { body, acceptSse: true });
    }
    return this.client.request("POST", path, { body });
  }
}

class McpResource extends _Resource {
  listTools(): Promise<unknown[]> {
    return this.client.request("GET", "/v1/mcp/tools");
  }
}

class SkillsResource extends _Resource {
  list(params: {
    q?: string;
    team_id?: string;
    cursor?: string;
    limit?: number;
  } = {}): Promise<unknown> {
    return this.client.request("GET", "/v1/xct-skills", { params });
  }

  get(skillId: string): Promise<unknown> {
    return this.client.request("GET", `/v1/xct-skills/${encodeURIComponent(skillId)}`);
  }
}

class ChatCompletionsResource extends _Resource {
  create(payload: Record<string, unknown> & { stream?: false }): Promise<unknown>;
  create(payload: Record<string, unknown> & { stream: true }): AsyncIterable<unknown>;
  create(
    payload: Record<string, unknown> & { stream?: boolean },
  ): Promise<unknown> | AsyncIterable<unknown> {
    if (payload.stream) {
      return this.client.stream("POST", "/v1/chat/completions", {
        body: payload,
        acceptSse: true,
      });
    }
    return this.client.request("POST", "/v1/chat/completions", { body: payload });
  }
}

class ChatResource extends _Resource {
  readonly completions: ChatCompletionsResource;
  constructor(client: XctClient) {
    super(client);
    this.completions = new ChatCompletionsResource(client);
  }
}

// ============================================================================
// SSE / NDJSON parsers
// ============================================================================

async function* parseSse<T>(body: ReadableStream<Uint8Array>): AsyncIterable<T> {
  const decoder = new TextDecoder();
  const reader = body.getReader();
  let buffer = "";
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      while (true) {
        const idx = buffer.indexOf("\n\n");
        if (idx === -1) break;
        const eventBlock = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        const parsed = coalesceSseEvent<T>(eventBlock);
        if (parsed !== undefined) yield parsed;
      }
    }
    if (buffer.trim()) {
      const parsed = coalesceSseEvent<T>(buffer);
      if (parsed !== undefined) yield parsed;
    }
  } finally {
    reader.releaseLock();
  }
}

function coalesceSseEvent<T>(block: string): T | undefined {
  const dataParts = block
    .split("\n")
    .filter((ln) => ln.startsWith("data:"))
    .map((ln) => ln.slice("data:".length).trimStart());
  if (dataParts.length === 0) return undefined;
  const raw = dataParts.join("\n").trim();
  if (!raw || raw === "[DONE]") return undefined;
  try {
    return JSON.parse(raw) as T;
  } catch {
    return undefined;
  }
}

async function* parseNdjson<T>(body: ReadableStream<Uint8Array>): AsyncIterable<T> {
  const decoder = new TextDecoder();
  const reader = body.getReader();
  let buffer = "";
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;
        try {
          yield JSON.parse(trimmed) as T;
        } catch {
          /* drop malformed line */
        }
      }
    }
    if (buffer.trim()) {
      try {
        yield JSON.parse(buffer.trim()) as T;
      } catch {
        /* drop */
      }
    }
  } finally {
    reader.releaseLock();
  }
}

function _safeJson(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

// re-export for convenience
export { XctError, CapabilityNotFoundError };
