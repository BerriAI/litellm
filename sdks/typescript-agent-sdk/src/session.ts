/**
 * SessionHandle — a single VM session running under one agent.
 *
 * Each session is a long-lived sandbox (Phase 1: noop VM) that processes
 * one or more `Run`s sequentially. `send()` starts a new run; `followup()`
 * queues a message into the active run.
 */

import { requestJson, type ResolvedClient } from "./client/http.js";
import { Run } from "./run.js";
import {
  type ConversationTurn,
  type ListOptions,
  type ListResult,
  type RunInfo,
  type SDKImage,
  type SessionInfo,
  type SessionStatus,
} from "./types.js";

interface SendInput {
  text: string;
  images?: SDKImage[];
}

export class SessionHandle {
  readonly id: string;
  readonly agentId: string;

  private _status: SessionStatus;
  private readonly _client: ResolvedClient;

  constructor(info: SessionInfo, client: ResolvedClient) {
    this.id = info.id;
    this.agentId = info.agentId;
    this._status = info.status;
    this._client = client;
  }

  get status(): SessionStatus {
    return this._status;
  }

  /** Start a new run with `input`. Throws 409 if a run is already in flight. */
  async send(input: string | SendInput): Promise<Run> {
    const body = normalizeSendInput(input);
    const info = await requestJson<RunInfo>(this._client, {
      method: "POST",
      path: `/v2/sessions/${encodeURIComponent(this.id)}/runs`,
      body,
    });
    return new Run(info, this._client);
  }

  /** Queue a follow-up message into the active run. */
  async followup(message: string): Promise<void> {
    // Wire shape matches backend `FollowupCreate`: {prompt: {text: "..."}}.
    await requestJson<void>(this._client, {
      method: "POST",
      path: `/v2/sessions/${encodeURIComponent(this.id)}/followup`,
      body: { prompt: { text: message } },
    });
  }

  /** Fetch a single run by ID. */
  async getRun(runId: string): Promise<Run> {
    const info = await requestJson<RunInfo>(this._client, {
      method: "GET",
      path: `/v2/sessions/${encodeURIComponent(this.id)}/runs/${encodeURIComponent(runId)}`,
    });
    return new Run(info, this._client);
  }

  /** List runs belonging to this session. */
  async listRuns(options: ListOptions = {}): Promise<ListResult<Run>> {
    const data = await requestJson<{ items: RunInfo[]; nextCursor?: string }>(
      this._client,
      {
        method: "GET",
        path: `/v2/sessions/${encodeURIComponent(this.id)}/runs`,
        query: { limit: options.limit, cursor: options.cursor },
      },
    );
    return {
      items: (data.items ?? []).map((info) => new Run(info, this._client)),
      nextCursor: data.nextCursor,
    };
  }

  /** Snapshot of the full conversation across runs. */
  async conversation(): Promise<ConversationTurn[]> {
    const data = await requestJson<{ turns: ConversationTurn[] }>(
      this._client,
      {
        method: "GET",
        path: `/v2/sessions/${encodeURIComponent(this.id)}/conversation`,
      },
    );
    return data.turns ?? [];
  }

  /** Tear down the VM and delete the session. */
  async delete(): Promise<void> {
    await requestJson<void>(this._client, {
      method: "DELETE",
      path: `/v2/sessions/${encodeURIComponent(this.id)}`,
    });
    this._status = "terminated";
  }

  /** Alias of `delete()`. */
  async terminate(): Promise<void> {
    await this.delete();
  }

  /**
   * Enables `await using session = await agent.createSession(...)`.
   * Calls DELETE on scope exit.
   */
  async [Symbol.asyncDispose](): Promise<void> {
    if (this._status !== "terminated") {
      try {
        await this.delete();
      } catch {
        // Best-effort cleanup — do not throw out of dispose.
      }
    }
  }
}

function normalizeSendInput(input: string | SendInput): {
  text: string;
  images: SDKImage[];
} {
  if (typeof input === "string") return { text: input, images: [] };
  return { text: input.text, images: input.images ?? [] };
}
