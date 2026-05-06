/**
 * Run — a single execution of an agent inside a session.
 *
 * Lifecycle: queued → running → completed | failed | cancelled.
 */

import { resolveClient, requestJson, type ResolvedClient } from "./client/http.js";
import { streamRunEvents } from "./client/sse.js";
import {
  type ConversationTurn,
  type RunEvent,
  type RunInfo,
  type RunResult,
  type RunStatus,
} from "./types.js";

const TERMINAL_STATES: RunStatus[] = ["completed", "failed", "cancelled"];

export class Run {
  readonly id: string;
  readonly sessionId: string;

  private _status: RunStatus;
  private _result: string | null;
  private _git: RunInfo["git"];
  private readonly _client: ResolvedClient;

  constructor(info: RunInfo, client: ResolvedClient) {
    this.id = info.id;
    this.sessionId = info.sessionId;
    this._status = info.status;
    this._result = info.result;
    this._git = info.git;
    this._client = client;
  }

  get status(): RunStatus {
    return this._status;
  }

  get result(): string | null {
    return this._result;
  }

  get git(): RunInfo["git"] {
    return this._git;
  }

  /** Open an SSE stream of events for this run. */
  stream(opts: { startingSeq?: number; signal?: AbortSignal } = {}): AsyncIterable<RunEvent> {
    return streamRunEvents(this._client, this.sessionId, this.id, opts);
  }

  /** Block until the run reaches a terminal status. */
  async wait(): Promise<RunResult> {
    while (!TERMINAL_STATES.includes(this._status)) {
      const info = await requestJson<RunInfo>(this._client, {
        method: "GET",
        path: `/v1/sessions/${encodeURIComponent(this.sessionId)}/runs/${encodeURIComponent(this.id)}`,
      });
      this._status = info.status;
      this._result = info.result;
      this._git = info.git;
      if (TERMINAL_STATES.includes(this._status)) break;
      await sleep(500);
    }
    return {
      id: this.id,
      status: this._status,
      result: this._result,
      git: this._git,
    };
  }

  /** Snapshot of the conversation up through this run. */
  async conversation(): Promise<ConversationTurn[]> {
    const data = await requestJson<{ turns: ConversationTurn[] }>(this._client, {
      method: "GET",
      path: `/v1/sessions/${encodeURIComponent(this.sessionId)}/runs/${encodeURIComponent(this.id)}/conversation`,
    });
    return data.turns ?? [];
  }

  /** Cancel a running run; no-op if already terminal. */
  async cancel(): Promise<void> {
    if (TERMINAL_STATES.includes(this._status)) return;
    await requestJson<void>(this._client, {
      method: "POST",
      path: `/v1/sessions/${encodeURIComponent(this.sessionId)}/runs/${encodeURIComponent(this.id)}/cancel`,
    });
    this._status = "cancelled";
  }
}

/** Internal helper used by SessionHandle / Agent to build a Run from a wire response. */
export function runFromInfo(info: RunInfo, options: { apiKey?: string; baseUrl?: string }): Run {
  return new Run(info, resolveClient(options));
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}
