import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import type { PropsWithChildren } from "react";
import { describe, expect, it, vi } from "vitest";

import { TracesApiContext, type TracesApi } from "../api";
import type { TraceSignals, TraceSummary } from "../types";
import { signalPollInterval, useTraceSignals } from "./useTraceSignals";

const run = (trace_id: string): TraceSummary => ({ trace_id, trace_ref: "" }) as TraceSummary;

const signals = (trace_id: string, status: TraceSignals["status"]): TraceSignals => ({
  trace_id,
  trace_ref: "",
  status,
  flags: status === "classified" ? [{ signal_id: "tool_failure", name: "Tool failure", score: 0.9 }] : [],
  model: "jev",
  classified_at: null,
});

describe("signalPollInterval", () => {
  it("polls fast only while a run is still waiting for a result", () => {
    const settled = signalPollInterval([signals("a", "classified"), signals("b", "failed")]);
    expect(signalPollInterval([signals("a", "classified"), signals("b", "unclassified")])).toBeLessThan(settled);
    expect(signalPollInterval([signals("a", "pending")])).toBeLessThan(settled);
    expect(signalPollInterval(undefined)).toBe(settled);
  });
});

describe("useTraceSignals", () => {
  it("keeps showing known results while a new run is added to the list", async () => {
    const gate = { release: (): void => undefined };
    const api = {
      live: true,
      signals: vi.fn(async (traces: { trace_id: string }[]) => {
        if (traces.length > 1) await new Promise<void>((resolve) => (gate.release = resolve));
        return traces.map((trace) => signals(trace.trace_id, "classified"));
      }),
    } as unknown as TracesApi;
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const wrapper = ({ children }: PropsWithChildren) => (
      <QueryClientProvider client={client}>
        <TracesApiContext.Provider value={api}>{children}</TracesApiContext.Provider>
      </QueryClientProvider>
    );
    const { result, rerender } = renderHook(({ runs }) => useTraceSignals("token", runs, true), {
      wrapper,
      initialProps: { runs: [run("old")] },
    });
    await waitFor(() => expect(result.current.get("old")?.status).toBe("ready"));

    rerender({ runs: [run("new"), run("old")] });

    expect(result.current.get("old")?.status).toBe("ready");
    expect(result.current.get("new")?.status).toBe("pending");
    gate.release();
    await waitFor(() => expect(result.current.get("new")?.status).toBe("ready"));
  });
});
