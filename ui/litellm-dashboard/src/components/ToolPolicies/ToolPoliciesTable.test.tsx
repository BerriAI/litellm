import React from "react";
import { describe, expect, it, vi } from "vitest";
import { fireEvent, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { OnUrlUpdateFunction } from "nuqs/adapters/testing";

import { chooseSelectOption, renderWithProviders } from "../../../tests/test-utils";
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

const SHUFFLED_TOOLS: ToolRow[] = [TOOLS[1], TOOLS[2], TOOLS[0]];

const SAME_TIME_TOOLS: ToolRow[] = SHUFFLED_TOOLS.map((tool) => ({ ...tool, created_at: "2026-07-20T10:00:00Z" }));

const renderTable = (
  overrides: Partial<React.ComponentProps<typeof ToolPoliciesTable>> = {},
  urlOptions: Parameters<typeof renderWithProviders>[1] = {},
) => {
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
  renderWithProviders(<ToolPoliciesTable {...props} />, urlOptions);
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
    renderTable({ data: SHUFFLED_TOOLS });

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

    fireEvent.change(screen.getByTestId("datatable-search"), { target: { value: "weather" } });

    await waitFor(() => expect(rowIds()).toEqual(["tool-1"]));
  });

  it("should match on key hash", async () => {
    const user = userEvent.setup();
    renderTable();

    fireEvent.change(screen.getByTestId("datatable-search"), { target: { value: "hash-bbb" } });

    await waitFor(() => expect(rowIds()).toEqual(["tool-2"]));
  });

  it("should not match on user agent", async () => {
    const user = userEvent.setup();
    renderTable();

    fireEvent.change(screen.getByTestId("datatable-search"), { target: { value: "curl" } });

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

  it.each([
    ["filter-input-policy", "All Input Policies"],
    ["filter-output-policy", "All Output Policies"],
    ["filter-team", "All Teams"],
    ["filter-key-alias", "All Keys"],
  ])("should show the human label on the %s trigger while unfiltered", async (testId, label) => {
    const user = userEvent.setup();
    renderTable();

    await user.click(screen.getByTestId("datatable-filters-trigger"));

    expect(await screen.findByTestId(testId)).toHaveTextContent(label);
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

const renderWithUrl = (searchParams: string, data: ToolRow[] = TOOLS) => {
  const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
  renderTable({ data }, { searchParams, onUrlUpdate });
  return onUrlUpdate;
};

const lastUrlUpdate = (onUrlUpdate: ReturnType<typeof vi.fn<OnUrlUpdateFunction>>) =>
  onUrlUpdate.mock.calls.at(-1)?.[0];

const MANY_TOOLS: ToolRow[] = Array.from({ length: 55 }, (_, index) => ({
  tool_id: `bulk-${String(index).padStart(2, "0")}`,
  tool_name: `bulk_tool_${index}`,
  input_policy: "untrusted",
  output_policy: "untrusted",
  call_count: index,
  key_hash: `hash-${index}`,
  created_at: new Date(Date.UTC(2026, 6, 1, 0, 0, 55 - index)).toISOString(),
}));

describe("ToolPoliciesTable URL state", () => {
  it("applies the search from the URL", () => {
    renderWithUrl("?search=weather");

    expect(rowIds()).toEqual(["tool-1"]);
    expect(screen.getByTestId("datatable-search")).toHaveValue("weather");
  });

  it("writes the search to the URL", async () => {
    const onUrlUpdate = renderWithUrl("");

    fireEvent.change(screen.getByTestId("datatable-search"), { target: { value: "search" } });

    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("search")).toBe("search"));
    expect(rowIds()).toEqual(["tool-2"]);
  });

  it.each([
    ["filter_input_policy=blocked", "input_policy", ["tool-3"]],
    ["filter_output_policy=trusted", "output_policy", ["tool-2"]],
    ["filter_team_id=team-alpha", "team_id", ["tool-1"]],
    ["filter_key_alias=dev-key", "key_alias", ["tool-2"]],
  ])("applies ?%s as a column filter", (query, columnId, expected) => {
    renderWithUrl(`?${query}`);

    expect(rowIds()).toEqual(expected);
    expect(screen.getByTestId(`filter-chip-${columnId}`)).toBeInTheDocument();
  });

  it("writes the filters applied in the drawer to the URL", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = renderWithUrl("");

    await user.click(screen.getByTestId("datatable-filters-trigger"));
    await pickFilter(user, "filter-output-policy", "untrusted");
    await pickFilter(user, "filter-key-alias", "prod-key");
    await user.click(screen.getByTestId("filter-drawer-apply"));

    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("filter_output_policy")).toBe("untrusted"));
    expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("filter_key_alias")).toBe("prod-key");
    expect(rowIds()).toEqual(["tool-1"]);
  });

  it("drops a filter from the URL when its chip is removed", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = renderWithUrl("?filter_team_id=team-alpha");

    await user.click(screen.getByTestId("filter-chip-remove-team_id"));

    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("filter_team_id")).toBe(false));
    expect(rowIds()).toEqual(["tool-1", "tool-2", "tool-3"]);
  });

  it("orders rows by the sort column and direction in the URL", () => {
    renderWithUrl("?sort_by=call_count&sort_order=asc");

    expect(rowIds()).toEqual(["tool-2", "tool-1", "tool-3"]);
  });

  it.each([
    ["input_policy", "asc", ["tool-3", "tool-2", "tool-1"]],
    ["output_policy", "desc", ["tool-3", "tool-1", "tool-2"]],
    ["team_id", "asc", ["tool-3", "tool-1", "tool-2"]],
    ["key_alias", "asc", ["tool-3", "tool-2", "tool-1"]],
  ])("orders rows by ?sort_by=%s&sort_order=%s", (column, order, expected) => {
    renderWithUrl(`?sort_by=${column}&sort_order=${order}`, SAME_TIME_TOOLS);

    expect(rowIds()).toEqual(expected);
  });

  it("falls back to newest first for an unknown sort column", () => {
    renderWithUrl("?sort_by=user_agent&sort_order=desc", SHUFFLED_TOOLS);

    expect(rowIds()).toEqual(["tool-1", "tool-2", "tool-3"]);
  });

  it("writes the sort to the URL when a header is clicked", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = renderWithUrl("");

    await user.click(screen.getByTestId("sort-header-tool_name"));

    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("sort_by")).toBe("tool_name"));
    expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("sort_order")).toBe("asc");
  });

  it("drops the page from the URL when the sort changes", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = renderWithUrl("?page=2", MANY_TOOLS);

    await user.click(screen.getByTestId("sort-header-tool_name"));

    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("sort_by")).toBe("tool_name"));
    expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("page")).toBe(false);
    expect(screen.getByTestId("pagination-page")).toHaveTextContent("Page 1 of 2");
  });

  it("drops the page from the URL when the search changes", async () => {
    const onUrlUpdate = renderWithUrl("?page=2", MANY_TOOLS);

    fireEvent.change(screen.getByTestId("datatable-search"), { target: { value: "bulk_tool" } });

    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("search")).toBe("bulk_tool"));
    expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("page")).toBe(false);
    expect(screen.getByTestId("pagination-page")).toHaveTextContent("Page 1 of 2");
  });

  it("drops the page from the URL when a drawer filter is applied", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = renderWithUrl("?page=2", MANY_TOOLS);

    await user.click(screen.getByTestId("datatable-filters-trigger"));
    await pickFilter(user, "filter-input-policy", "untrusted");
    await user.click(screen.getByTestId("filter-drawer-apply"));

    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("filter_input_policy")).toBe("untrusted"));
    expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("page")).toBe(false);
    expect(screen.getByTestId("pagination-page")).toHaveTextContent("Page 1 of 2");
  });

  it("opens the page named in the URL", () => {
    renderWithUrl("?page=2", MANY_TOOLS);

    expect(screen.getByTestId("pagination-page")).toHaveTextContent("Page 2 of 2");
    expect(rowIds()).toEqual(["bulk-50", "bulk-51", "bulk-52", "bulk-53", "bulk-54"]);
  });

  it("writes the page to the URL when paging forward", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = renderWithUrl("", MANY_TOOLS);

    await user.click(screen.getByRole("button", { name: "Go to next page" }));

    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("page")).toBe("2"));
    expect(rowIds()[0]).toBe("bulk-50");
  });

  it("uses the page size from the URL", () => {
    renderWithUrl("?page_size=100", MANY_TOOLS);

    expect(rowIds()).toHaveLength(55);
    expect(screen.getByTestId("pagination-page")).toHaveTextContent("Page 1 of 1");
  });

  it("writes the page size chosen in the pager to the URL", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = renderWithUrl("", MANY_TOOLS);

    await chooseSelectOption(user, screen.getByTestId("pagination-page-size"), "100");

    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("page_size")).toBe("100"));
    expect(rowIds()).toHaveLength(55);
  });
});
