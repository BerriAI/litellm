import { screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { render } from "../../../../tests/test-utils";
import type { LogEntry } from "../columns";
import { DrawerHeader } from "./DrawerHeader";

const logEntry = (overrides: Partial<LogEntry>): LogEntry =>
  ({
    request_id: "170d64ea-69f0-431a-be72-332f8f78c18a",
    api_key: "key-1",
    team_id: "team-1",
    model: "gpt-4o",
    model_id: "model-1",
    custom_llm_provider: "openai",
    call_type: "acompletion",
    spend: 0.01,
    total_tokens: 10,
    prompt_tokens: 5,
    completion_tokens: 5,
    startTime: "2026-07-07T09:50:13Z",
    endTime: "2026-07-07T09:50:14Z",
    cache_hit: "false",
    messages: [],
    response: {},
    ...overrides,
  }) as LogEntry;

const renderHeader = (log: LogEntry, isSidebarCollapsed: boolean) =>
  render(
    <DrawerHeader
      log={log}
      onClose={vi.fn()}
      onPrevious={vi.fn()}
      onNext={vi.fn()}
      isSidebarCollapsed={isSidebarCollapsed}
      onToggleSidebar={vi.fn()}
      statusLabel="Failure"
      statusColor="error"
      environment="default"
    />,
  );

const expandToggle = () => screen.getByLabelText("Expand trace sidebar");

describe("DrawerHeader sidebar toggle", () => {
  it("stays out of the header while the sidebar owns it", () => {
    renderHeader(logEntry({}), false);

    expect(screen.queryByLabelText("Expand trace sidebar")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Collapse trace sidebar")).not.toBeInTheDocument();
  });

  it("shares the model row once the sidebar is collapsed", () => {
    renderHeader(logEntry({}), true);

    const row = expandToggle().parentElement as HTMLElement;
    expect(within(row).getByText("gpt-4o")).toBeInTheDocument();
  });

  it("falls back to the request id row when the log names no model", () => {
    renderHeader(logEntry({ model: "", custom_llm_provider: "" }), true);

    const row = expandToggle().parentElement as HTMLElement;
    expect(within(row).getByText("170d64ea-69f0-431a-be72-332f8f78c18a")).toBeInTheDocument();
  });
});
