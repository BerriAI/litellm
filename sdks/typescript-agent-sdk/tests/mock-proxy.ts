/**
 * In-memory mock proxy for SDK tests.
 *
 * We don't use msw here because we need to test SSE streaming + mid-stream
 * socket drops, which fetch interception libraries handle awkwardly.
 * Instead we spin up a real `http.Server` on a random port and point the SDK
 * at it.
 */

import http, { type IncomingMessage, type ServerResponse } from "node:http";
import { AddressInfo } from "node:net";

interface AgentRecord {
  id: string;
  name: string;
  model: { id: string };
  systemPrompt?: string;
  metadata: Record<string, string>;
  createdAt: string;
  sessions: Map<string, SessionRecord>;
}

interface SessionRecord {
  id: string;
  agentId: string;
  status: "provisioning" | "ready" | "busy" | "error" | "terminated";
  vmId: string;
  createdAt: string;
  runs: Map<string, RunRecord>;
  followups: string[];
  conversation: {
    role: string;
    content: string;
    runId?: string;
    createdAt: string;
  }[];
}

interface RunRecord {
  id: string;
  sessionId: string;
  status: "queued" | "running" | "finished" | "cancelled" | "error";
  result: string | null;
  events: { seq: number; type: string; data: unknown }[];
  startedAt: string | null;
  completedAt: string | null;
  /** active SSE connections — set when /events is connected */
  sseClients: Set<ServerResponse>;
}

export interface MockProxyOptions {
  /**
   * If set, the first SSE connection for a run is killed after this many ms,
   * to simulate a mid-stream socket drop. Reset to `false` after the first
   * drop to let reconnection succeed.
   */
  killSseAfterMs?: number;
  /** If true, emit a 503 with retry-after on the first request. */
  failFirstRequest?: boolean;
}

export class MockProxy {
  readonly server: http.Server;
  readonly agents: Map<string, AgentRecord> = new Map();
  private firstRequestServed = false;
  private firstStreamDropped = false;
  private idCounter = 0;
  options: MockProxyOptions;

  constructor(options: MockProxyOptions = {}) {
    this.options = options;
    this.server = http.createServer((req, res) => this.handle(req, res));
  }

  async start(): Promise<string> {
    await new Promise<void>((resolve) =>
      this.server.listen(0, "127.0.0.1", resolve),
    );
    const addr = this.server.address() as AddressInfo;
    return `http://127.0.0.1:${addr.port}`;
  }

  async stop(): Promise<void> {
    // Force-close any open SSE clients first.
    for (const agent of this.agents.values()) {
      for (const session of agent.sessions.values()) {
        for (const run of session.runs.values()) {
          for (const c of run.sseClients) c.end();
          run.sseClients.clear();
        }
      }
    }
    await new Promise<void>((resolve, reject) =>
      this.server.close((err) => (err ? reject(err) : resolve())),
    );
  }

  /** Lookup helpers exposed for tests. */
  findRun(runId: string): RunRecord | undefined {
    for (const agent of this.agents.values()) {
      for (const session of agent.sessions.values()) {
        const run = session.runs.get(runId);
        if (run) return run;
      }
    }
    return undefined;
  }

  findSession(sessionId: string): SessionRecord | undefined {
    for (const agent of this.agents.values()) {
      const s = agent.sessions.get(sessionId);
      if (s) return s;
    }
    return undefined;
  }

  /** Append a new event to a run, fan it out to live SSE clients. */
  emit(runId: string, type: string, data: unknown): void {
    const run = this.findRun(runId);
    if (!run) return;
    const seq = run.events.length;
    const event = { seq, type, data };
    run.events.push(event);
    const payload = `id: ${seq}\ndata: ${JSON.stringify(event)}\n\n`;
    for (const client of run.sseClients) {
      client.write(payload);
    }
  }

  /** Mark a run completed and emit a `done` event. */
  complete(runId: string, result: string): void {
    const run = this.findRun(runId);
    if (!run) return;
    run.status = "finished";
    run.result = result;
    run.completedAt = new Date().toISOString();
    this.emit(runId, "done", { result });
    for (const client of run.sseClients) client.end();
    run.sseClients.clear();
  }

  private nextId(prefix: string): string {
    this.idCounter++;
    return `${prefix}_${this.idCounter}`;
  }

  private async handle(
    req: IncomingMessage,
    res: ServerResponse,
  ): Promise<void> {
    if (this.options.failFirstRequest && !this.firstRequestServed) {
      this.firstRequestServed = true;
      res.writeHead(503, {
        "retry-after": "0",
        "content-type": "application/json",
      });
      res.end(
        JSON.stringify({ error: { code: "transient", message: "boom" } }),
      );
      return;
    }
    this.firstRequestServed = true;

    if (req.headers.authorization !== "Bearer test-key") {
      res.writeHead(401, { "content-type": "application/json" });
      res.end(
        JSON.stringify({ error: { code: "unauthorized", message: "bad key" } }),
      );
      return;
    }

    const url = new URL(req.url ?? "/", "http://localhost");
    const method = req.method ?? "GET";
    const segments = url.pathname.split("/").filter(Boolean);

    try {
      if (segments[0] !== "v2") {
        return notFound(res);
      }
      // /v2/agents
      if (segments.length === 2 && segments[1] === "agents") {
        if (method === "POST") return this.createAgent(req, res);
        if (method === "GET") return this.listAgents(res);
      }
      if (segments.length === 3 && segments[1] === "agents") {
        const agent = this.agents.get(segments[2]);
        if (!agent) return notFound(res);
        if (method === "GET") return ok(res, this.serializeAgent(agent));
        if (method === "DELETE") {
          this.agents.delete(agent.id);
          return ok(res, { ok: true });
        }
      }
      if (
        segments.length === 4 &&
        segments[1] === "agents" &&
        segments[3] === "sessions"
      ) {
        const agent = this.agents.get(segments[2]);
        if (!agent) return notFound(res);
        if (method === "POST") return this.createSession(agent, req, res);
        if (method === "GET")
          return ok(res, {
            items: [...agent.sessions.values()].map(serializeSession),
          });
      }
      if (
        segments.length === 5 &&
        segments[1] === "agents" &&
        segments[3] === "sessions"
      ) {
        const agent = this.agents.get(segments[2]);
        const session = agent?.sessions.get(segments[4]);
        if (!session) return notFound(res);
        if (method === "GET") return ok(res, serializeSession(session));
      }
      // /v2/sessions/{sid}/...
      if (segments[1] === "sessions" && segments.length >= 3) {
        const session = this.findSession(segments[2]);
        if (!session) return notFound(res);

        if (segments.length === 3) {
          if (method === "DELETE") {
            session.status = "terminated";
            return ok(res, { ok: true });
          }
        }
        if (
          segments.length === 4 &&
          segments[3] === "runs" &&
          method === "POST"
        ) {
          return this.createRun(session, req, res);
        }
        if (
          segments.length === 4 &&
          segments[3] === "runs" &&
          method === "GET"
        ) {
          return ok(res, {
            items: [...session.runs.values()].map(serializeRun),
          });
        }
        if (
          segments.length === 4 &&
          segments[3] === "followup" &&
          method === "POST"
        ) {
          const body = await readJson(req);
          // Wire format: {prompt: {text: "..."}} per FollowupCreate.
          const text = body?.prompt?.text ?? "";
          session.followups.push(text);
          session.conversation.push({
            role: "user",
            content: text,
            createdAt: new Date().toISOString(),
          });
          return ok(res, { ok: true });
        }
        if (
          segments.length === 4 &&
          segments[3] === "conversation" &&
          method === "GET"
        ) {
          return ok(res, { turns: session.conversation.map(serializeTurn) });
        }
        if (segments.length === 5 && segments[3] === "runs") {
          const run = session.runs.get(segments[4]);
          if (!run) return notFound(res);
          if (method === "GET") return ok(res, serializeRun(run));
        }
        if (
          segments.length === 6 &&
          segments[3] === "runs" &&
          segments[5] === "events"
        ) {
          const run = session.runs.get(segments[4]);
          if (!run) return notFound(res);
          return this.streamEvents(run, req, res, url);
        }
        if (
          segments.length === 6 &&
          segments[3] === "runs" &&
          segments[5] === "conversation"
        ) {
          const run = session.runs.get(segments[4]);
          if (!run) return notFound(res);
          return ok(res, {
            turns: session.conversation
              .filter((t) => !t.runId || t.runId === run.id)
              .map(serializeTurn),
          });
        }
        if (
          segments.length === 6 &&
          segments[3] === "runs" &&
          segments[5] === "cancel" &&
          method === "POST"
        ) {
          const run = session.runs.get(segments[4]);
          if (!run) return notFound(res);
          run.status = "cancelled";
          run.completedAt = new Date().toISOString();
          return ok(res, { ok: true });
        }
      }
      return notFound(res);
    } catch (e) {
      res.writeHead(500, { "content-type": "application/json" });
      res.end(
        JSON.stringify({
          error: { code: "internal", message: (e as Error).message },
        }),
      );
    }
  }

  private async createAgent(
    req: IncomingMessage,
    res: ServerResponse,
  ): Promise<void> {
    const body = await readJson(req);
    const id = this.nextId("agt");
    const agent: AgentRecord = {
      id,
      name: body.name ?? "unnamed",
      model: body.model ?? { id: "test-model" },
      // Wire format is snake_case (Python idiom); SDK transforms camelCase
      // public API to snake_case before sending.
      systemPrompt: body.system_prompt,
      metadata: body.metadata ?? {},
      createdAt: new Date().toISOString(),
      sessions: new Map(),
    };
    this.agents.set(id, agent);
    return ok(res, this.serializeAgent(agent));
  }

  private listAgents(res: ServerResponse): void {
    return ok(res, {
      items: [...this.agents.values()].map((a) => this.serializeAgent(a)),
    });
  }

  private async createSession(
    agent: AgentRecord,
    req: IncomingMessage,
    res: ServerResponse,
  ): Promise<void> {
    await readJson(req);
    const id = this.nextId("ses");
    const vmId = this.nextId("vm");
    const session: SessionRecord = {
      id,
      agentId: agent.id,
      status: "ready",
      vmId,
      createdAt: new Date().toISOString(),
      runs: new Map(),
      followups: [],
      conversation: [],
    };
    agent.sessions.set(id, session);
    return ok(res, serializeSession(session));
  }

  private async createRun(
    session: SessionRecord,
    req: IncomingMessage,
    res: ServerResponse,
  ): Promise<void> {
    const body = await readJson(req);
    // 409 if any non-terminal run exists.
    for (const r of session.runs.values()) {
      if (r.status === "queued" || r.status === "running") {
        res.writeHead(409, { "content-type": "application/json" });
        res.end(
          JSON.stringify({
            error: { code: "session_busy", message: "run in flight" },
          }),
        );
        return;
      }
    }
    const id = this.nextId("run");
    const run: RunRecord = {
      id,
      sessionId: session.id,
      status: "queued",
      result: null,
      events: [],
      startedAt: new Date().toISOString(),
      completedAt: null,
      sseClients: new Set(),
    };
    session.runs.set(id, run);
    session.conversation.push({
      role: "user",
      content: body.text ?? "",
      runId: id,
      createdAt: new Date().toISOString(),
    });
    return ok(res, serializeRun(run));
  }

  private streamEvents(
    run: RunRecord,
    req: IncomingMessage,
    res: ServerResponse,
    url: URL,
  ): void {
    const startingSeq = Number(url.searchParams.get("starting_seq") ?? -1);
    res.writeHead(200, {
      "content-type": "text/event-stream",
      "cache-control": "no-cache",
      connection: "keep-alive",
    });

    // Replay events from startingSeq forward.
    const replayFrom =
      Number.isFinite(startingSeq) && startingSeq >= 0 ? startingSeq : 0;
    for (const event of run.events) {
      if (event.seq < replayFrom) continue;
      res.write(`id: ${event.seq}\ndata: ${JSON.stringify(event)}\n\n`);
    }

    if (
      run.status === "finished" ||
      run.status === "error" ||
      run.status === "cancelled"
    ) {
      res.end();
      return;
    }

    run.sseClients.add(res);
    const drop = !this.firstStreamDropped && this.options.killSseAfterMs;
    if (drop) {
      this.firstStreamDropped = true;
      setTimeout(() => {
        // Force-close socket without flushing any final event.
        req.socket.destroy();
        run.sseClients.delete(res);
      }, this.options.killSseAfterMs);
    }
    req.on("close", () => {
      run.sseClients.delete(res);
    });
  }

  private serializeAgent(agent: AgentRecord) {
    // Wire format is snake_case; SDK transforms back to camelCase on receive.
    return {
      id: agent.id,
      name: agent.name,
      model: agent.model,
      created_at: agent.createdAt,
    };
  }
}

function serializeSession(s: SessionRecord) {
  return {
    id: s.id,
    agent_id: s.agentId,
    status: s.status,
    created_at: s.createdAt,
  };
}

function serializeRun(r: RunRecord) {
  return {
    id: r.id,
    session_id: r.sessionId,
    status: r.status,
    result: r.result,
    started_at: r.startedAt,
    completed_at: r.completedAt,
  };
}

function serializeTurn(t: {
  role: string;
  content: string;
  runId?: string;
  createdAt: string;
}) {
  return {
    role: t.role,
    content: t.content,
    run_id: t.runId,
    created_at: t.createdAt,
  };
}

function ok(res: ServerResponse, body: unknown): void {
  res.writeHead(200, { "content-type": "application/json" });
  res.end(JSON.stringify(body));
}

function notFound(res: ServerResponse): void {
  res.writeHead(404, { "content-type": "application/json" });
  res.end(
    JSON.stringify({ error: { code: "not_found", message: "not found" } }),
  );
}

function readJson(req: IncomingMessage): Promise<any> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    req.on("data", (c) => chunks.push(c));
    req.on("end", () => {
      const raw = Buffer.concat(chunks).toString();
      if (!raw) return resolve({});
      try {
        resolve(JSON.parse(raw));
      } catch (e) {
        reject(e);
      }
    });
    req.on("error", reject);
  });
}
