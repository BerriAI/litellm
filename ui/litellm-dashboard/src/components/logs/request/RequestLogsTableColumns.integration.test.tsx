import { render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { DataTable } from "@/components/shared/DataTable";

import type { LogEntry } from "../types";
import { getRequestLogsTableColumns } from "./RequestLogsTableColumns";

const logEntry = (overrides: Partial<LogEntry>): LogEntry => ({
  request_id: "request",
  api_key: "key",
  team_id: "team",
  model: "test-model",
  model_id: "model",
  call_type: "acompletion",
  spend: 0,
  total_tokens: 10,
  prompt_tokens: 5,
  completion_tokens: 5,
  startTime: "2026-01-01T00:00:00Z",
  endTime: "2026-01-01T00:00:01Z",
  cache_hit: "None",
  messages: [],
  response: {},
  ...overrides,
});

function renderCacheCells(logs: LogEntry[]) {
  render(
    <DataTable
      data={logs}
      columns={getRequestLogsTableColumns({ onKeyHashClick: vi.fn(), onSessionClick: vi.fn() })}
      getRowId={(row) => row.request_id}
    />,
  );
  const header = screen.getByRole("columnheader", { name: "Cache Hit" });
  const columnIndex = screen.getAllByRole("columnheader").indexOf(header);
  return screen
    .getAllByRole("row")
    .slice(1)
    .map((row) => within(row).getAllByRole("cell")[columnIndex]);
}

describe("Request log cache hits", () => {
  it("distinguishes a response cache hit, an enabled cache miss and caching disabled", () => {
    const cells = renderCacheCells([
      logEntry({ request_id: "hit", cache_hit: "True" }),
      logEntry({ request_id: "miss", cache_key: "response-cache-key" }),
      logEntry({ request_id: "off", cache_key: "Cache OFF" }),
    ]);

    expect(cells[0]).toHaveTextContent(/^Yes$/);
    expect(cells[1]).toHaveTextContent(/^No$/);
    expect(cells[2]).toHaveTextContent(/^-$/);
  });

  it("shows session hit counts instead of the representative request's cache status", () => {
    const sessions: Partial<LogEntry>[] = [
      { request_id: "mixed", cache_hit: "False", session_total_count: 3, session_cache_hit_count: 2 },
      { request_id: "one-hit", cache_hit: "False", session_total_count: 2, session_cache_hit_count: 1 },
      { request_id: "no-hits", session_total_count: 2, session_cache_hit_count: 0 },
      { request_id: "unknown", cache_hit: "True", session_total_count: 2 },
    ];
    const cells = renderCacheCells(sessions.map(logEntry));

    expect(cells[0]).toHaveTextContent(/^2 hits$/);
    expect(cells[1]).toHaveTextContent(/^1 hit$/);
    expect(cells[2]).toHaveTextContent(/^0 hits$/);
    expect(cells[3]).toHaveTextContent(/^-$/);
  });

  it("does not report provider prompt caching as a response cache hit", () => {
    const promptCached: Partial<LogEntry> = {
      cache_key: "Cache OFF",
      session_total_count: 1,
      session_cache_hit_count: 0,
      metadata: { additional_usage_values: { cache_read_input_tokens: 100 } },
    };
    const [cell] = renderCacheCells([logEntry(promptCached)]);

    expect(cell).toHaveTextContent(/^-$/);
  });
});
