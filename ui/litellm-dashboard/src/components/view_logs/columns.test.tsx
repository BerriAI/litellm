import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { createColumns, type LogEntry } from "./columns";
import { DataTable } from "./table";

const logEntry = (overrides: Partial<LogEntry>): LogEntry => ({
  request_id: "req-1",
  api_key: "key-1",
  team_id: "team-1",
  model: "gpt-4o",
  model_id: "model-1",
  call_type: "acompletion",
  spend: 0,
  total_tokens: 10,
  prompt_tokens: 5,
  completion_tokens: 5,
  startTime: "2026-07-07T09:50:13Z",
  endTime: "2026-07-07T09:50:14Z",
  cache_hit: "false",
  messages: [],
  response: {},
  ...overrides,
});

describe("Cost column", () => {
  it("renders '-' for zero spend with no tooltip, so hovering never shows a contradictory $0", async () => {
    const user = userEvent.setup();
    render(
      <DataTable
        data={[logEntry({ request_id: "req-zero", spend: 0 })]}
        columns={createColumns()}
        getRowId={(r) => r.request_id}
      />,
    );
    for (const dash of screen.getAllByText("-")) {
      await user.hover(dash);
    }
    expect(screen.queryByText("$0")).not.toBeInTheDocument();
  });

  it("shows the full-precision raw value in the tooltip for a real spend", async () => {
    const user = userEvent.setup();
    render(
      <DataTable
        data={[logEntry({ request_id: "req-spend", spend: 0.00012345678 })]}
        columns={createColumns()}
        getRowId={(r) => r.request_id}
      />,
    );
    const formatted = screen.getByText("$0.000123");
    await user.hover(formatted);
    expect(await screen.findByText("$0.00012345678")).toBeInTheDocument();
  });
});
