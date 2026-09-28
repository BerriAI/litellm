import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, within } from "@testing-library/react";
import React from "react";
import { describe, expect, it, vi } from "vitest";

import { FairnessStatusTable } from "./FairnessStatusTable";
import type { FairnessStatusResponse, WorkloadClassStatus } from "./schema";

const CLASS: WorkloadClassStatus = {
  name: "production",
  reserved_share: 0.5,
  reserved_rpm: 50,
  reserved_tpm: null,
  current_requests: 12,
  current_tokens: 3400,
  queue_depth: 2,
  max_queue_wait_seconds: 30,
  queued_total: 7,
  admitted_after_wait_total: 5,
  rejected_capacity_total: 1,
  rejected_queue_full_total: 2,
  rejected_deadline_total: 3,
  disconnected_total: 4,
  avg_queue_wait_seconds: 1.234,
};

const STATUS: FairnessStatusResponse = {
  enabled: true,
  limiter_active: true,
  saturation_threshold: 0.5,
  window_size_seconds: 60,
  stats_window_seconds: 300,
  models: [
    {
      model_group: "gpt-5.2",
      rpm: 100,
      tpm: null,
      saturation: 0.72,
      enforcing_reservations: true,
      current_requests: 72,
      current_tokens: 9000,
      classes: [CLASS, { ...CLASS, name: "default", reserved_share: 0.25, reserved_rpm: 25, queue_depth: 0 }],
    },
  ],
};

const renderTable = (status: FairnessStatusResponse) => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={queryClient}>
      <FairnessStatusTable fetchStatus={vi.fn().mockResolvedValue(status)} refreshIntervalMs={false} />
    </QueryClientProvider>,
  );
};

describe("FairnessStatusTable", () => {
  it("shows saturation, enforcement state and per-class queue and rejection telemetry", async () => {
    renderTable(STATUS);

    const model = await screen.findByTestId("model-status-gpt-5.2");
    expect(model).toHaveTextContent("saturated, reservations enforced");
    expect(model).toHaveTextContent("72% of 100 rpm / no limit tpm (threshold 50%)");

    const row = within(screen.getByTestId("class-row-production"));
    expect(row.getByText("50%")).toBeInTheDocument();
    expect(row.getByText("50 rpm / no limit tpm")).toBeInTheDocument();
    expect(row.getByText("12 req / 3,400 tok")).toBeInTheDocument();
    expect(row.getByText("deadline 30s")).toBeInTheDocument();
    expect(row.getByText("7 queued, 5 admitted")).toBeInTheDocument();
    expect(row.getByText("avg wait 1.23s")).toBeInTheDocument();
    expect(row.getByText("1 capacity, 2 queue full, 3 deadline, 4 disconnected")).toBeInTheDocument();
    expect(row.getByText("10")).toBeInTheDocument();
    expect(screen.getByTestId("class-row-default")).toBeInTheDocument();
  });

  it("explains that nothing is collected while the limiter is off", async () => {
    const inactive: FairnessStatusResponse = { ...STATUS, enabled: false, limiter_active: false, models: [] };
    renderTable(inactive);

    expect(await screen.findByText("disabled")).toBeInTheDocument();
    expect(screen.getByText(/Enable fairness and save/)).toBeInTheDocument();
  });

  it("marks a model as borrowing when it is below the threshold", async () => {
    renderTable({
      ...STATUS,
      models: [{ ...STATUS.models[0], saturation: 0.1, enforcing_reservations: false }],
    });

    expect(await screen.findByText("borrowing allowed")).toBeInTheDocument();
  });
});
