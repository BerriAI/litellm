import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { DataTable } from "@/components/shared/DataTable";

import type { LogEntry } from "./columns";
import { getRequestLogsTableColumns } from "./RequestLogsTableColumns";

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

const noopDeps = { onKeyHashClick: vi.fn(), onSessionClick: vi.fn() };

function renderRows(rows: LogEntry[], deps = noopDeps) {
  render(
    <DataTable
      data={rows}
      columns={getRequestLogsTableColumns(deps)}
      getRowId={(row) => row.request_id}
      size="compact"
    />,
  );
}

describe("Cost column", () => {
  it("renders '-' for zero spend with no tooltip, so hovering never shows a contradictory $0", async () => {
    const user = userEvent.setup();
    renderRows([logEntry({ request_id: "req-zero", spend: 0 })]);

    for (const dash of screen.getAllByText("-")) {
      await user.hover(dash);
    }
    expect(screen.queryByText("$0")).not.toBeInTheDocument();
  });

  it("shows the full-precision raw value in the tooltip for a real spend", async () => {
    const user = userEvent.setup();
    renderRows([logEntry({ request_id: "req-spend", spend: 0.00012345678 })]);

    await user.hover(screen.getByText("$0.000123"));
    expect(await screen.findByText("$0.00012345678")).toBeInTheDocument();
  });

  it("shows the summed session total, not the representative call's spend, for a multi-round session", () => {
    renderRows([
      logEntry({
        request_id: "req-session",
        spend: 0.01,
        session_id: "sess-1",
        session_total_count: 3,
        session_total_spend: 0.06,
      }),
    ]);

    expect(screen.getByText("$0.060000")).toBeInTheDocument();
    expect(screen.queryByText("$0.010000")).not.toBeInTheDocument();
    expect(screen.getByText("session total")).toBeInTheDocument();
  });
});

describe("row action cells", () => {
  it("reports the key hash through the injected dependency rather than a row field", async () => {
    const user = userEvent.setup();
    const deps = { onKeyHashClick: vi.fn(), onSessionClick: vi.fn() };
    renderRows([logEntry({ request_id: "req-key", metadata: { user_api_key: "sk-hash-9" } })], deps);

    await user.click(screen.getByText("sk-hash-9"));
    expect(deps.onKeyHashClick).toHaveBeenCalledWith("sk-hash-9");
  });

  it("reports the session id from the session cell", async () => {
    const user = userEvent.setup();
    const deps = { onKeyHashClick: vi.fn(), onSessionClick: vi.fn() };
    renderRows([logEntry({ request_id: "req-sess", session_id: "sess-42" })], deps);

    await user.click(screen.getByText("sess-42"));
    expect(deps.onSessionClick).toHaveBeenCalledWith("sess-42");
  });
});

describe("sortable headers", () => {
  it("exposes sort controls only for the backend-sortable fields", () => {
    renderRows([logEntry({})]);

    for (const field of ["startTime", "spend", "request_duration_ms", "ttft_ms", "model", "total_tokens"]) {
      expect(screen.getByTestId(`sort-trigger-${field}`)).toBeInTheDocument();
    }
    for (const field of ["request_id", "session_id", "status", "type", "end_user"]) {
      expect(screen.queryByTestId(`sort-trigger-${field}`)).toBeNull();
    }
  });
});

describe("TTFT column", () => {
  it("renders '-' when the completion start equals the end time, since TTFT is meaningless there", () => {
    renderRows([
      logEntry({
        request_id: "req-nonstream",
        endTime: "2026-07-07T09:50:14Z",
        completionStartTime: "2026-07-07T09:50:14Z",
      }),
    ]);

    expect(screen.queryByText("1.00")).not.toBeInTheDocument();
  });

  it("renders seconds when streaming produced a real first token", () => {
    renderRows([
      logEntry({
        request_id: "req-stream",
        startTime: "2026-07-07T09:50:13Z",
        endTime: "2026-07-07T09:50:16Z",
        completionStartTime: "2026-07-07T09:50:14Z",
      }),
    ]);

    expect(screen.getByText("1.00")).toBeInTheDocument();
  });
});
