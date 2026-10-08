import { screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders, testQueryClient } from "../../../../../tests/test-utils";
import { type TracesApi, TracesApiContext } from "../api";
import type { TraceSummary } from "../types";
import { useTraceFeedback } from "./useTraceFeedback";

const run = { trace_id: "trace-1", trace_ref: "REF1" } as TraceSummary;

function Probe() {
  const state = useTraceFeedback("sk-test", [run], true).get("REF1");
  return <output aria-label="feedback state">{state?.status ?? "missing"}</output>;
}

const renderWith = (feedbackSummary: () => Promise<unknown>) =>
  renderWithProviders(
    <TracesApiContext.Provider value={{ live: false, feedbackSummary } as unknown as TracesApi}>
      <Probe />
    </TracesApiContext.Provider>,
  );

describe("useTraceFeedback", () => {
  beforeEach(() => testQueryClient.clear());

  it("reads a run's summary from the batch response", async () => {
    renderWith(async () => [{ ...run, count: 1, average: 2, lowest: 2 }]);
    expect(await screen.findByText("ready")).toBeInTheDocument();
  });

  it("marks feedback unavailable instead of crashing when the proxy answers with something other than a list", async () => {
    renderWith(vi.fn(async () => ({ detail: "Not Found" })));
    expect(await screen.findByText("error")).toBeInTheDocument();
  });
});
