/**
 * SSE client with auto-reconnect via `Last-Event-ID` / `?starting_seq=N`.
 *
 * Yields parsed `RunEvent` objects. When the stream drops, it transparently
 * reconnects from the last seq it saw. Honors AbortSignal.
 */

import { createParser, type EventSourceMessage } from "eventsource-parser";
import { LiteLLMAgentError, type RunEvent } from "../types.js";
import { request, type ResolvedClient } from "./http.js";

export interface StreamRunOptions {
  startingSeq?: number;
  signal?: AbortSignal;
  /** Max reconnect attempts on dropped sockets. Default: 5. */
  maxReconnects?: number;
}

/**
 * Stream events from `GET /v2/sessions/{sid}/runs/{rid}/events`.
 *
 * Resume contract:
 *  - On every event, we record `seq`.
 *  - On reconnect, we send both `Last-Event-ID: <seq>` (per the SSE spec)
 *    and `?starting_seq=<seq+1>` (LiteLLM-specific). The proxy honors
 *    whichever it understands and replays from there.
 */
export async function* streamRunEvents(
  client: ResolvedClient,
  sessionId: string,
  runId: string,
  opts: StreamRunOptions = {},
): AsyncIterable<RunEvent> {
  const maxReconnects = opts.maxReconnects ?? 5;
  let lastSeq = opts.startingSeq !== undefined ? opts.startingSeq - 1 : -1;
  let reconnects = 0;

  while (true) {
    const startingSeq = lastSeq >= 0 ? lastSeq + 1 : undefined;
    let res: Response;
    try {
      res = await request(client, {
        method: "GET",
        path: `/v2/sessions/${encodeURIComponent(sessionId)}/runs/${encodeURIComponent(runId)}/events`,
        query:
          startingSeq !== undefined ? { starting_seq: startingSeq } : undefined,
        headers:
          startingSeq !== undefined ? { "Last-Event-ID": String(lastSeq) } : {},
        stream: true,
        signal: opts.signal,
      });
    } catch (e) {
      if (opts.signal?.aborted) return;
      if (reconnects >= maxReconnects) throw e;
      reconnects++;
      await sleep(backoffMs(reconnects));
      continue;
    }

    if (!res.body) {
      throw new LiteLLMAgentError("SSE response had no body", {
        code: "stream_no_body",
      });
    }

    let droppedMidStream = false;
    let sawProgressOnThisConnection = false;
    try {
      for await (const event of parseSSE(res.body, opts.signal)) {
        const parsed = decodeEvent(event);
        if (parsed === null) continue;
        if (parsed.seq <= lastSeq) {
          // Replayed event we already saw; drop it.
          continue;
        }
        lastSeq = parsed.seq;
        sawProgressOnThisConnection = true;
        yield parsed;
        if (parsed.type === "done" || parsed.type === "error") {
          return;
        }
      }
      // Stream ended cleanly without a `done` event — server hung up.
      droppedMidStream = true;
    } catch (e) {
      if (opts.signal?.aborted) return;
      droppedMidStream = true;
    }

    if (!droppedMidStream) return;
    // Reset the reconnect budget if this connection delivered new events.
    // The counter tracks *consecutive* failures, not lifetime drops, so a
    // long-running stream with sporadic transient drops is not penalized.
    if (sawProgressOnThisConnection) {
      reconnects = 0;
    }
    if (reconnects >= maxReconnects) {
      throw new LiteLLMAgentError(
        `SSE reconnect budget exhausted (${maxReconnects})`,
        { code: "sse_reconnect_exhausted" },
      );
    }
    reconnects++;
    await sleep(backoffMs(reconnects));
  }
}

function decodeEvent(msg: EventSourceMessage): RunEvent | null {
  if (!msg.data) return null;
  try {
    const obj = JSON.parse(msg.data) as Partial<RunEvent>;
    if (typeof obj.seq !== "number" || typeof obj.type !== "string") {
      return null;
    }
    return {
      seq: obj.seq,
      type: obj.type,
      data: obj.data ?? null,
    };
  } catch {
    return null;
  }
}

async function* parseSSE(
  body: ReadableStream<Uint8Array>,
  signal?: AbortSignal,
): AsyncIterable<EventSourceMessage> {
  const queue: EventSourceMessage[] = [];
  const state: { resolve: (() => void) | null; done: boolean } = {
    resolve: null,
    done: false,
  };

  const parser = createParser({
    onEvent: (e) => {
      queue.push(e);
      state.resolve?.();
    },
  });

  const reader = body.getReader();
  const decoder = new TextDecoder();

  const pump = (async () => {
    try {
      while (true) {
        if (signal?.aborted) {
          await reader.cancel();
          return;
        }
        const { value, done: rdone } = await reader.read();
        if (rdone) return;
        parser.feed(decoder.decode(value, { stream: true }));
        state.resolve?.();
      }
    } finally {
      state.done = true;
      state.resolve?.();
    }
  })();

  try {
    while (true) {
      while (queue.length > 0) {
        const ev = queue.shift()!;
        yield ev;
      }
      if (state.done) return;
      await new Promise<void>((r) => {
        state.resolve = r;
      });
      state.resolve = null;
    }
  } finally {
    await pump.catch(() => {});
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

function backoffMs(attempt: number): number {
  return Math.min(5_000, 250 * Math.pow(2, attempt));
}
