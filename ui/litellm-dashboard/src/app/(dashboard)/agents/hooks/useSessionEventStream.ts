"use client";

/**
 * useSessionEventStream — SSE hook with auto-reconnect and seq-cursor resume.
 *
 * Subscribes to the active Run's event stream for a Session and surfaces a
 * deduped, monotonically-ordered list of `CloudAgentRunEvent` to the consumer.
 *
 * Reconnect strategy:
 *   - On `error`, close the EventSource and re-open after backoff.
 *   - When re-opening, pass `since_seq=<lastSeq>` so the server can replay
 *     missed events. Dedup on the client by checking `seq > lastSeq`.
 *   - On `online` (browser-level reconnect), retry immediately.
 *
 * Mock mode (`NEXT_PUBLIC_USE_MOCK_AGENTS=true`):
 *   - Replays `MOCK_RUN_EVENTS` with a 400ms cadence so the UI looks live.
 *   - Replays from `sinceSeq` when reconnecting, so the dedup path is testable.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { buildRunEventStreamUrl } from "@/lib/cloud-agents-client";
import { isMockEnabled, MOCK_RUN_EVENTS } from "@/lib/mock-agents";
import type { CloudAgentRunEvent } from "@/types/cloud-agents";

const MOCK_TICK_MS = 400;
const RECONNECT_BACKOFF_MS = 1000;

interface UseSessionEventStreamArgs {
  sessionId: string | null;
  runId: string | null;
  enabled?: boolean;
}

interface UseSessionEventStreamResult {
  events: CloudAgentRunEvent[];
  connected: boolean;
  lastSeq: number;
  error: string | null;
}

export function useSessionEventStream({
  sessionId,
  runId,
  enabled = true,
}: UseSessionEventStreamArgs): UseSessionEventStreamResult {
  const [events, setEvents] = useState<CloudAgentRunEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const lastSeqRef = useRef(0);
  const esRef = useRef<EventSource | null>(null);
  const mockTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const mockOfflineRef = useRef(false);

  const appendEvent = useCallback((evt: CloudAgentRunEvent) => {
    if (evt.seq <= lastSeqRef.current) {
      return; // dedup: server replayed an event we already have
    }
    lastSeqRef.current = evt.seq;
    setEvents((prev) => [...prev, evt]);
  }, []);

  // Mock streamer — replays canned events from the next unseen seq.
  const startMockStream = useCallback(() => {
    setConnected(true);
    let i = MOCK_RUN_EVENTS.findIndex((e) => e.seq > lastSeqRef.current);
    if (i < 0) i = MOCK_RUN_EVENTS.length;
    const tick = () => {
      if (mockOfflineRef.current) {
        // simulated offline: stop emitting until brought back online
        mockTimerRef.current = setTimeout(tick, MOCK_TICK_MS);
        return;
      }
      if (i >= MOCK_RUN_EVENTS.length) {
        return; // exhausted; leave connected=true to mimic an idle stream
      }
      appendEvent(MOCK_RUN_EVENTS[i]);
      i += 1;
      mockTimerRef.current = setTimeout(tick, MOCK_TICK_MS);
    };
    mockTimerRef.current = setTimeout(tick, MOCK_TICK_MS);
  }, [appendEvent]);

  const stopMockStream = useCallback(() => {
    if (mockTimerRef.current) {
      clearTimeout(mockTimerRef.current);
      mockTimerRef.current = null;
    }
    setConnected(false);
  }, []);

  // Real EventSource streamer with reconnect+resume.
  const startRealStream = useCallback(() => {
    if (!sessionId || !runId) return;

    const url = buildRunEventStreamUrl(sessionId, runId, lastSeqRef.current || undefined);
    const es = new EventSource(url, { withCredentials: true });
    esRef.current = es;

    es.onopen = () => {
      setConnected(true);
      setError(null);
    };
    es.onmessage = (msg: MessageEvent<string>) => {
      try {
        const evt = JSON.parse(msg.data) as CloudAgentRunEvent;
        if (typeof evt.seq === "number") {
          appendEvent(evt);
        }
      } catch (e) {
        // ignore malformed events; the server is responsible for shape
      }
    };
    es.onerror = () => {
      setConnected(false);
      setError("stream interrupted; reconnecting");
      es.close();
      esRef.current = null;
      // Reconnect after backoff, replaying from lastSeq.
      setTimeout(() => {
        if (esRef.current === null) startRealStream();
      }, RECONNECT_BACKOFF_MS);
    };
  }, [sessionId, runId, appendEvent]);

  const stopRealStream = useCallback(() => {
    if (esRef.current) {
      esRef.current.close();
      esRef.current = null;
    }
    setConnected(false);
  }, []);

  // Expose offline/online for Playwright reconnect testing in mock mode.
  useEffect(() => {
    if (!isMockEnabled()) return;
    const onOffline = () => {
      mockOfflineRef.current = true;
      setConnected(false);
    };
    const onOnline = () => {
      mockOfflineRef.current = false;
      setConnected(true);
    };
    window.addEventListener("offline", onOffline);
    window.addEventListener("online", onOnline);
    return () => {
      window.removeEventListener("offline", onOffline);
      window.removeEventListener("online", onOnline);
    };
  }, []);

  useEffect(() => {
    if (!enabled || !sessionId || !runId) return;
    if (isMockEnabled()) {
      startMockStream();
      return () => stopMockStream();
    }
    startRealStream();
    return () => stopRealStream();
  }, [enabled, sessionId, runId, startMockStream, stopMockStream, startRealStream, stopRealStream]);

  return { events, connected, lastSeq: lastSeqRef.current, error };
}
