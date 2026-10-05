import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "@/../tests/test-utils";
import type { Lens } from "../../model/types";
import { InvestigationActions, type InvestigationActionsProps } from "./InvestigationActions";

const lens: Lens = {
  version: 0,
  spent: 0,
  id: "lens",
  scope: { all_teams: true, api_key_hash: "", team_id: "" },
  settings: {
    context: "",
    source: "traces",
    lookback_hours: 24,
    service: "",
    agent_name: "",
    filters: [],
    interval_minutes: 15,
    sample_size: 100,
    sample_percent: 100,
    concurrency: 8,
    team_id: "",
    execution_ids: [],
    monthly_budget: 20,
    name: "Release reviews",
    model: "analysis",
    enabled: true,
    checks: [],
  },
  revision: 1,
  created_at: "2026-09-30T10:00:00Z",
  next_run_at: "2026-09-30T10:00:00Z",
  budget_month: "2026-09",
  findings: [],
  jobs: [],
};

function renderActions(overrides: Partial<InvestigationActionsProps> = {}) {
  const intents = {
    onEdit: vi.fn(),
    onDuplicate: vi.fn(),
    onPause: vi.fn(),
    onEnableMonitoring: vi.fn(),
    onRunNow: vi.fn(),
  };
  renderWithProviders(<InvestigationActions lens={lens} ready busy={false} {...intents} {...overrides} />);
  return intents;
}

describe("InvestigationActions", () => {
  it("reports each menu choice as an intent without touching the API", async () => {
    const user = userEvent.setup();
    const intents = renderActions();
    await user.click(screen.getByRole("button", { name: "Run now" }));
    expect(intents.onRunNow).toHaveBeenCalledOnce();
    await user.click(screen.getByRole("button", { name: "Investigation actions" }));
    await user.click(await screen.findByRole("menuitem", { name: "Pause monitoring" }));
    expect(intents.onPause).toHaveBeenCalledOnce();
    expect(intents.onEnableMonitoring).not.toHaveBeenCalled();
    await user.click(screen.getByRole("button", { name: "Investigation actions" }));
    await user.click(await screen.findByRole("menuitem", { name: "Edit investigation" }));
    expect(intents.onEdit).toHaveBeenCalledOnce();
    await user.click(screen.getByRole("button", { name: "Investigation actions" }));
    await user.click(await screen.findByRole("menuitem", { name: "Duplicate" }));
    expect(intents.onDuplicate).toHaveBeenCalledOnce();
  });

  it("offers to enable monitoring on a paused investigation", async () => {
    const user = userEvent.setup();
    const paused = { ...lens, settings: { ...lens.settings, enabled: false } };
    const intents = renderActions({ lens: paused });
    await user.click(screen.getByRole("button", { name: "Investigation actions" }));
    await user.click(await screen.findByRole("menuitem", { name: "Enable monitoring" }));
    expect(intents.onEnableMonitoring).toHaveBeenCalledOnce();
    expect(intents.onPause).not.toHaveBeenCalled();
  });

  it("blocks a second run while one is queued or running", () => {
    const job = {
      id: "job",
      status: "queued" as const,
      stage: "Queued",
      trigger: "manual" as const,
      attempts: 0,
      error: "",
      cost: 0,
      created_at: lens.created_at,
      start: lens.created_at,
      end: lens.created_at,
      revision: 1,
      settings: lens.settings,
      assessments: [],
      steps: [],
      findings: [],
    };
    renderActions({ lens: { ...lens, jobs: [job] } });
    expect(screen.getByRole("button", { name: "Run now" })).toBeDisabled();
  });
});
