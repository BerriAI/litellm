import { describe, it, expect, vi } from "vitest";
import { render } from "@testing-library/react";
import React from "react";
import { createColumns, LogEntry } from "./columns";
import type { LogsSortField } from "./columns";

vi.mock("../provider_info_helpers", () => ({
  getProviderLogoAndName: (provider: string) => ({
    logo: `https://logo.test/${provider}.png`,
    displayName: provider,
  }),
}));

const makeLogEntry = (overrides: Partial<LogEntry> = {}): LogEntry => ({
  request_id: "req-1",
  api_key: "sk-test",
  team_id: "team-1",
  model: "gpt-4",
  model_id: "gpt-4-id",
  call_type: "completion",
  spend: 0.005,
  total_tokens: 150,
  prompt_tokens: 100,
  completion_tokens: 50,
  startTime: "2025-01-15T10:00:00Z",
  endTime: "2025-01-15T10:00:02Z",
  cache_hit: "none",
  messages: [],
  response: [],
  ...overrides,
});

const renderCell = (columnKey: string, row: LogEntry) => {
  const cols = createColumns();
  const col = cols.find(
    (c: any) => c.accessorKey === columnKey || c.id === columnKey,
  );
  if (!col || !col.cell) throw new Error(`Column "${columnKey}" not found`);

  const nestedValue = columnKey.includes(".")
    ? columnKey.split(".").reduce((obj: any, key: string) => obj?.[key], row)
    : (row as any)[columnKey];

  const cellContext: any = {
    getValue: () => nestedValue,
    row: { original: row, getIsExpanded: () => false, getCanExpand: () => false, getToggleExpandedHandler: () => () => {} },
  };

  const CellComponent = typeof col.cell === "function" ? col.cell : () => null;
  const { container } = render(<>{CellComponent(cellContext)}</>);
  return container;
};

// ---------------------------------------------------------------------------
// createColumns — sort prop behavior
// ---------------------------------------------------------------------------
describe("createColumns", () => {
  it("uses plain string headers when no sortProps provided", () => {
    const cols = createColumns();
    const timeCol = cols.find((c: any) => c.accessorKey === "startTime");
    expect(timeCol?.header).toBe("Time");
  });

  it("uses SortableHeader render functions when sortProps provided", () => {
    const cols = createColumns({
      sortBy: "startTime" as LogsSortField,
      sortOrder: "desc",
      onSortChange: vi.fn(),
    });
    const timeCol = cols.find((c: any) => c.accessorKey === "startTime");
    expect(typeof timeCol?.header).toBe("function");
  });
});

// ---------------------------------------------------------------------------
// Duration column — ms to seconds conversion
// ---------------------------------------------------------------------------
describe("Duration column", () => {
  it("converts ms to seconds with 2 decimal places", () => {
    const container = renderCell("request_duration_ms", makeLogEntry({ request_duration_ms: 1234 }));
    expect(container.textContent).toContain("1.23");
  });

  it('renders "-" when duration is null', () => {
    const container = renderCell("request_duration_ms", makeLogEntry({ request_duration_ms: undefined }));
    expect(container.textContent).toBe("-");
  });

  it("does not treat 0 as null", () => {
    const container = renderCell("request_duration_ms", makeLogEntry({ request_duration_ms: 0 }));
    expect(container.textContent).toContain("0.00");
  });
});

// ---------------------------------------------------------------------------
// TTFT column — timestamp math with edge cases
// ---------------------------------------------------------------------------
describe("TTFT column", () => {
  it("computes TTFT from startTime and completionStartTime", () => {
    const container = renderCell("completionStartTime", makeLogEntry({
      startTime: "2025-01-15T10:00:00.000Z",
      endTime: "2025-01-15T10:00:03.000Z",
      completionStartTime: "2025-01-15T10:00:01.500Z",
    }));
    expect(container.textContent).toContain("1.50");
  });

  it('renders "-" when completionStartTime is null', () => {
    const container = renderCell("completionStartTime", makeLogEntry({ completionStartTime: undefined }));
    expect(container.textContent).toBe("-");
  });

  it('renders "-" when completionStartTime equals endTime (non-streaming)', () => {
    const container = renderCell("completionStartTime", makeLogEntry({
      endTime: "2025-01-15T10:00:02Z",
      completionStartTime: "2025-01-15T10:00:02Z",
    }));
    expect(container.textContent).toBe("-");
  });

  it('renders "-" when TTFT would be negative', () => {
    const container = renderCell("completionStartTime", makeLogEntry({
      startTime: "2025-01-15T10:00:05.000Z",
      endTime: "2025-01-15T10:00:06.000Z",
      completionStartTime: "2025-01-15T10:00:04.000Z",
    }));
    expect(container.textContent).toBe("-");
  });
});

// ---------------------------------------------------------------------------
// Tags column — overflow count logic
// ---------------------------------------------------------------------------
describe("Tags column", () => {
  it('renders "-" for empty tags', () => {
    const container = renderCell("request_tags", makeLogEntry({ request_tags: {} }));
    expect(container.textContent).toBe("-");
  });

  it('renders "-" for undefined tags', () => {
    const container = renderCell("request_tags", makeLogEntry({ request_tags: undefined }));
    expect(container.textContent).toBe("-");
  });

  it("renders single tag without +N suffix", () => {
    const container = renderCell("request_tags", makeLogEntry({ request_tags: { env: "prod" } }));
    expect(container.textContent).toContain("env: prod");
    expect(container.textContent).not.toContain("+");
  });

  it("renders first tag with +N for additional tags", () => {
    const container = renderCell("request_tags", makeLogEntry({
      request_tags: { env: "prod", team: "ml", version: "2" },
    }));
    expect(container.textContent).toContain("env: prod");
    expect(container.textContent).toContain("+2");
  });
});

// ---------------------------------------------------------------------------
// Model column — logo fallback chain
// ---------------------------------------------------------------------------
describe("Model column", () => {
  it("uses default provider logo", () => {
    const container = renderCell("model", makeLogEntry({ model: "claude-3", custom_llm_provider: "anthropic" }));
    const img = container.querySelector("img");
    expect(img?.getAttribute("src")).toContain("anthropic");
  });

  it("prefers MCP server logo over default provider logo", () => {
    const container = renderCell("model", makeLogEntry({
      model: "tool-model",
      custom_llm_provider: "openai",
      metadata: {
        mcp_tool_call_metadata: { mcp_server_logo_url: "https://custom-logo.test/mcp.png" },
      },
    }));
    const img = container.querySelector("img");
    expect(img?.getAttribute("src")).toBe("https://custom-logo.test/mcp.png");
  });
});
