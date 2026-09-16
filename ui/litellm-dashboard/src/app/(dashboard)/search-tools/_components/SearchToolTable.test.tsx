import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "@/../tests/test-utils";
import SearchToolTable from "./SearchToolTable";
import { AvailableSearchProvider, SearchTool } from "./types";

const makeSearchTool = (overrides: Partial<SearchTool> = {}): SearchTool => ({
  search_tool_id: "tool-1",
  search_tool_name: "Perplexity Search",
  litellm_params: {
    search_provider: "perplexity",
  },
  created_at: "2024-01-15T10:30:00Z",
  updated_at: "2024-01-16T10:30:00Z",
  ...overrides,
});

const availableProviders: AvailableSearchProvider[] = [
  { provider_name: "perplexity", ui_friendly_name: "Perplexity AI" },
];

const defaultProps = {
  searchTools: [makeSearchTool()],
  isLoading: false,
  availableProviders,
  onView: vi.fn(),
  onEdit: vi.fn(),
  onDelete: vi.fn(),
};

const namedTools = (count: number): SearchTool[] =>
  Array.from({ length: count }, (_, index) =>
    makeSearchTool({
      search_tool_id: `tool-${index}`,
      search_tool_name: `tool-name-${String(index).padStart(2, "0")}`,
      created_at: new Date(Date.UTC(2024, 0, 1 + index)).toISOString(),
    }),
  );

const firstRowName = () => within(screen.getAllByRole("row")[1]).getByText(/^tool-name-/).textContent;

describe("SearchToolTable", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should display search tool information with the friendly provider name", () => {
    renderWithProviders(<SearchToolTable {...defaultProps} />);
    expect(screen.getByText("Perplexity Search")).toBeInTheDocument();
    expect(screen.getByText("tool-1")).toBeInTheDocument();
    expect(screen.getByText("Perplexity AI")).toBeInTheDocument();
    expect(screen.getByText("DB")).toBeInTheDocument();
  });

  it("should call onView when the search tool ID is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(<SearchToolTable {...defaultProps} />);
    await user.click(screen.getByRole("button", { name: /tool-1/ }));
    expect(defaultProps.onView).toHaveBeenCalledWith("tool-1");
  });

  it("should call onEdit and onDelete from the actions menu for a DB tool", async () => {
    const user = userEvent.setup();
    renderWithProviders(<SearchToolTable {...defaultProps} />);

    await user.click(screen.getByTestId("search-tool-actions-tool-1"));
    await user.click(await screen.findByTestId("search-tool-action-edit"));
    expect(defaultProps.onEdit).toHaveBeenCalledWith("tool-1");

    await user.click(screen.getByTestId("search-tool-actions-tool-1"));
    await user.click(await screen.findByTestId("search-tool-action-delete"));
    expect(defaultProps.onDelete).toHaveBeenCalledWith("tool-1");
  });

  it("should show a dash instead of a clickable ID for config tools", () => {
    const configTool = makeSearchTool({ search_tool_id: "config-tool", is_from_config: true });
    renderWithProviders(<SearchToolTable {...defaultProps} searchTools={[configTool]} />);
    expect(screen.queryByRole("button", { name: /config-tool/ })).not.toBeInTheDocument();
  });

  it("should disable Edit and Delete for config tools and suppress their callbacks", async () => {
    const user = userEvent.setup();
    const configTool = makeSearchTool({ search_tool_id: "config-tool", is_from_config: true });
    renderWithProviders(<SearchToolTable {...defaultProps} searchTools={[configTool]} />);

    await user.click(screen.getByTestId("search-tool-actions-config-tool"));

    const editItem = await screen.findByTestId("search-tool-action-edit");
    const deleteItem = await screen.findByTestId("search-tool-action-delete");
    expect(editItem).toHaveAttribute("data-disabled");
    expect(deleteItem).toHaveAttribute("data-disabled");

    await user.click(editItem);
    await user.click(deleteItem);
    expect(defaultProps.onEdit).not.toHaveBeenCalled();
    expect(defaultProps.onDelete).not.toHaveBeenCalled();
  });

  it("should sort tools by created_at descending by default", () => {
    const tools = [
      makeSearchTool({
        search_tool_id: "tool-old",
        search_tool_name: "older-tool",
        created_at: "2024-01-01T00:00:00Z",
      }),
      makeSearchTool({
        search_tool_id: "tool-new",
        search_tool_name: "newer-tool",
        created_at: "2024-06-01T00:00:00Z",
      }),
    ];
    renderWithProviders(<SearchToolTable {...defaultProps} searchTools={tools} />);
    const rows = screen.getAllByRole("row").slice(1);
    expect(within(rows[0]).getByText("newer-tool")).toBeInTheDocument();
    expect(within(rows[1]).getByText("older-tool")).toBeInTheDocument();
  });

  it("should show skeleton rows when loading", () => {
    renderWithProviders(<SearchToolTable {...defaultProps} searchTools={[]} isLoading />);
    expect(screen.getAllByTestId("skeleton-row").length).toBeGreaterThan(0);
  });

  it("should show the empty state when there are no search tools", () => {
    renderWithProviders(<SearchToolTable {...defaultProps} searchTools={[]} />);
    expect(screen.getByText("No search tools configured")).toBeInTheDocument();
  });

  describe("URL state", () => {
    it("orders rows by the sort_by and sort_order in the URL", () => {
      renderWithProviders(<SearchToolTable {...defaultProps} searchTools={namedTools(3)} />, {
        searchParams: { sort_by: "search_tool_name", sort_order: "asc" },
      });
      expect(firstRowName()).toBe("tool-name-00");
    });

    it("opens on the page in the URL", () => {
      renderWithProviders(<SearchToolTable {...defaultProps} searchTools={namedTools(30)} />, {
        searchParams: { page: "2" },
      });
      expect(screen.getAllByRole("row")).toHaveLength(6);
      expect(firstRowName()).toBe("tool-name-04");
    });

    it("writes the clicked sort column to the URL", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderWithProviders(<SearchToolTable {...defaultProps} searchTools={namedTools(3)} />, { onUrlUpdate });

      await user.click(screen.getByTestId("sort-header-search_tool_name"));

      await waitFor(() => expect(onUrlUpdate).toHaveBeenCalled());
      const params = onUrlUpdate.mock.calls.at(-1)?.[0].searchParams;
      expect(params?.get("sort_by")).toBe("search_tool_name");
      expect(params?.get("sort_order")).toBe("asc");
      expect(firstRowName()).toBe("tool-name-00");
    });

    it("writes the next page to the URL", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderWithProviders(<SearchToolTable {...defaultProps} searchTools={namedTools(30)} />, { onUrlUpdate });
      expect(firstRowName()).toBe("tool-name-29");

      await user.click(screen.getByRole("button", { name: "Go to next page" }));

      await waitFor(() => expect(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.get("page")).toBe("2"));
      expect(firstRowName()).toBe("tool-name-04");
    });
  });
});
