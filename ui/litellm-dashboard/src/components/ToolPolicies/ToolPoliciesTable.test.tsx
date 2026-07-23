import React from "react";
import { describe, expect, it, vi } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { renderWithProviders } from "../../../tests/test-utils";
import type { ToolRow } from "@/components/networking";
import { ToolPoliciesTable } from "./ToolPoliciesTable";

const TOOLS: ToolRow[] = [
  {
    tool_id: "tool-1",
    tool_name: "get_weather",
    input_policy: "untrusted",
    output_policy: "untrusted",
    call_count: 12,
    team_id: "team-alpha",
    key_alias: "prod-key",
    key_hash: "hash-aaa",
    user_agent: "curl/8.7.1",
    created_at: "2026-07-21T10:00:00Z",
  },
  {
    tool_id: "tool-2",
    tool_name: "search_web",
    input_policy: "trusted",
    output_policy: "trusted",
    call_count: 5,
    team_id: "team-beta",
    key_alias: "dev-key",
    key_hash: "hash-bbb",
    created_at: "2026-07-20T10:00:00Z",
  },
  {
    tool_id: "tool-3",
    tool_name: "delete_file",
    input_policy: "blocked",
    output_policy: "untrusted",
    call_count: 100,
    key_hash: "hash-ccc",
    created_at: "2026-07-19T10:00:00Z",
  },
];

const renderTable = (overrides: Partial<React.ComponentProps<typeof ToolPoliciesTable>> = {}) => {
  const props = {
    data: TOOLS,
    isLoading: false,
    isRefreshing: false,
    onRefresh: vi.fn(),
    onSelectTool: vi.fn(),
    savingInput: new Set<string>(),
    savingOutput: new Set<string>(),
    onInputPolicyChange: vi.fn(),
    onOutputPolicyChange: vi.fn(),
    ...overrides,
  };
  renderWithProviders(<ToolPoliciesTable {...props} />);
  return props;
};

const rowIds = (): (string | null)[] =>
  Array.from(document.querySelectorAll("tbody tr[data-row-id]")).map((row) => row.getAttribute("data-row-id"));

const pickFilter = async (
  user: ReturnType<typeof userEvent.setup>,
  triggerTestId: string,
  optionLabel: string,
): Promise<void> => {
  await user.click(screen.getByTestId(triggerTestId));
  await user.click(await screen.findByRole("option", { name: optionLabel }));
};

describe("ToolPoliciesTable sorting", () => {
  it("should default to newest discovered first", () => {
    renderTable();

    expect(rowIds()).toEqual(["tool-1", "tool-2", "tool-3"]);
  });

  it("should sort by tool name when its header is used", async () => {
    const user = userEvent.setup();
    renderTable();

    await user.click(screen.getByTestId("sort-header-tool_name"));

    expect(rowIds()).toEqual(["tool-3", "tool-1", "tool-2"]);
  });
});

describe("ToolPoliciesTable search", () => {
  it("should match on tool name", async () => {
    const user = userEvent.setup();
    renderTable();

    await user.type(screen.getByTestId("datatable-search"), "weather");

    await waitFor(() => expect(rowIds()).toEqual(["tool-1"]));
  });

  it("should match on key hash", async () => {
    const user = userEvent.setup();
    renderTable();

    await user.type(screen.getByTestId("datatable-search"), "hash-bbb");

    await waitFor(() => expect(rowIds()).toEqual(["tool-2"]));
  });

  it("should not match on user agent", async () => {
    const user = userEvent.setup();
    renderTable();

    await user.type(screen.getByTestId("datatable-search"), "curl");

    await waitFor(() => expect(rowIds()).toEqual([]));
    expect(screen.getByText("No matching tools")).toBeInTheDocument();
  });
});

describe("ToolPoliciesTable filters", () => {
  it("should match an input policy exactly rather than as a substring", async () => {
    const user = userEvent.setup();
    renderTable();

    await user.click(screen.getByTestId("datatable-filters-trigger"));
    await pickFilter(user, "filter-input-policy", "trusted");
    await user.click(screen.getByTestId("filter-drawer-apply"));

    await waitFor(() => expect(rowIds()).toEqual(["tool-2"]));
  });

  it("should filter by team", async () => {
    const user = userEvent.setup();
    renderTable();

    await user.click(screen.getByTestId("datatable-filters-trigger"));
    await pickFilter(user, "filter-team", "team-alpha");
    await user.click(screen.getByTestId("filter-drawer-apply"));

    await waitFor(() => expect(rowIds()).toEqual(["tool-1"]));
    expect(screen.getByTestId("filter-chip-team_id")).toHaveTextContent("Team Name:");
  });

  it("should offer only the teams and keys present in the loaded rows", async () => {
    const user = userEvent.setup();
    renderTable();

    await user.click(screen.getByTestId("datatable-filters-trigger"));
    await user.click(screen.getByTestId("filter-team"));

    const teams = (await screen.findAllByRole("option")).map((option) => option.textContent);
    expect(teams).toEqual(["All Teams", "team-alpha", "team-beta"]);
  });
});

describe("ToolPoliciesTable chrome", () => {
  it("should open the detail view from the tool name cell", async () => {
    const user = userEvent.setup();
    const { onSelectTool } = renderTable();

    await user.click(screen.getByRole("button", { name: /get_weather/ }));

    expect(onSelectTool).toHaveBeenCalledWith("get_weather");
  });

  it("should refresh on demand", async () => {
    const user = userEvent.setup();
    const { onRefresh } = renderTable();

    await user.click(screen.getByTestId("datatable-refresh"));

    expect(onRefresh).toHaveBeenCalledTimes(1);
  });

  it("should explain how discovery works when there are no tools at all", () => {
    renderTable({ data: [] });

    expect(screen.getByText("No tools discovered")).toBeInTheDocument();
    expect(screen.getByText(/tool_calls to start auto-discovery/)).toBeInTheDocument();
  });

  it("should show skeleton rows while the first load is in flight", () => {
    renderTable({ data: [], isLoading: true });

    expect(screen.getAllByTestId("skeleton-row").length).toBeGreaterThan(0);
    expect(screen.queryByText("No tools discovered")).not.toBeInTheDocument();
  });
});
