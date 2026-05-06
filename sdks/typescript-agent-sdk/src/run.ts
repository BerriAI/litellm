/**
 * Run — a single execution of an agent inside a session.
 *
 * Lifecycle: queued → running → completed | failed | cancelled.
 */

import { requestJson, type ResolvedClient } from "./client/http.js";
import { streamRunEvents } from "./client/sse.js";
import {
  LiteLLMAgentError,
  type RunEvent,
  type RunInfo,
  type RunResult,
  type RunStatus,
} from "./types.js";

const TERMINAL_STATES: RunStatus[] = ["finished", "cancelled", "error"];
const DEFAULT_WAIT_POLL_MS = 500;

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
  stream(
    opts: { startingSeq?: number; signal?: AbortSignal } = {},
  ): AsyncIterable<RunEvent> {
    return streamRunEvents(this._client, this.sessionId, this.id, opts);
  }

  /**
   * Block until the run reaches a terminal status.
   *
   * @param opts.signal    Optional `AbortSignal` to bail out early. Throws
   *                       `LiteLLMAgentError({ code: "wait_aborted" })`.
   * @param opts.timeoutMs Optional ceiling on total wait time. Throws
   *                       `LiteLLMAgentError({ code: "wait_timeout" })`.
   * @param opts.pollMs    Override the poll interval (default 500ms).
   */
  async wait(
    opts: { signal?: AbortSignal; timeoutMs?: number; pollMs?: number } = {},
  ): Promise<RunResult> {
    const pollMs = opts.pollMs ?? DEFAULT_WAIT_POLL_MS;
    const deadline =
      opts.timeoutMs !== undefined && opts.timeoutMs > 0
        ? Date.now() + opts.timeoutMs
        : undefined;

    while (!TERMINAL_STATES.includes(this._status)) {
      if (opts.signal?.aborted) {
        throw new LiteLLMAgentError("wait() aborted", { code: "wait_aborted" });
      }
      if (deadline !== undefined && Date.now() >= deadline) {
        throw new LiteLLMAgentError(
          `wait() timed out after ${opts.timeoutMs}ms`,
          { code: "wait_timeout" },
        );
      }
      const info = await requestJson<RunInfo>(this._client, {
        method: "GET",
        path: `/v2/sessions/${encodeURIComponent(this.sessionId)}/runs/${encodeURIComponent(this.id)}`,
        signal: opts.signal,
      });
      this._status = info.status;
      this._result = info.result;
      this._git = info.git;
      if (TERMINAL_STATES.includes(this._status)) break;
      await sleep(pollMs, opts.signal);
    }
    return {
      id: this.id,
      status: this._status,
      result: this._result,
      git: this._git,
    };
  }

  // NOTE: There is no per-run conversation endpoint on the backend. The
  // conversation is session-scoped — call `session.conversation()` (on the
  // owning `SessionHandle`) for the full turn history.

  /** Cancel a running run; no-op if already terminal. */
  async cancel(): Promise<void> {
    if (TERMINAL_STATES.includes(this._status)) return;
    await requestJson<void>(this._client, {
      method: "POST",
      path: `/v2/sessions/${encodeURIComponent(this.sessionId)}/runs/${encodeURIComponent(this.id)}/cancel`,
    });
    this._status = "cancelled";
  }
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new LiteLLMAgentError("wait() aborted", { code: "wait_aborted" }));
      return;
    }
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    const onAbort = () => {
      clearTimeout(timer);
      reject(new LiteLLMAgentError("wait() aborted", { code: "wait_aborted" }));
    };
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}
