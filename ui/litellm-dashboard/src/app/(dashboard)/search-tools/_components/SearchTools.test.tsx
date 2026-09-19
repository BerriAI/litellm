import * as roles from "@/utils/roles";
import { QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { NuqsTestingAdapter, type OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders, testQueryClient } from "@/../tests/test-utils";
import * as networking from "@/components/networking";
import SearchTools from "./SearchTools";
import { AvailableSearchProvider, SearchTool } from "./types";

vi.mock("@/components/networking", () => ({
  fetchSearchTools: vi.fn(),
  updateSearchTool: vi.fn(),
  deleteSearchTool: vi.fn(),
  fetchAvailableSearchProviders: vi.fn(),
}));

vi.mock("@/utils/roles", () => ({
  isAdminRole: vi.fn(),
}));

const searchToolViewMounts = vi.hoisted(() => ({ count: 0 }));

vi.mock("./SearchToolView", async () => {
  const { useEffect } = await import("react");
  const SearchToolView = ({ searchTool, onBack }: { searchTool: SearchTool; onBack: () => void }) => {
    useEffect(() => {
      searchToolViewMounts.count += 1;
    }, []);
    return (
      <div data-testid="search-tool-view">
        <div>Search Tool View: {searchTool.search_tool_name}</div>
        <button onClick={onBack}>Back</button>
      </div>
    );
  };
  SearchToolView.displayName = "SearchToolView";
  return { SearchToolView };
});

vi.mock("./CreateSearchTools", () => {
  const CreateSearchTools = ({
    isModalVisible,
    setModalVisible,
  }: {
    isModalVisible: boolean;
    setModalVisible: (visible: boolean) => void;
  }) =>
    isModalVisible ? (
      <div data-testid="create-search-tool-modal">
        <button onClick={() => setModalVisible(false)}>Close Create Modal</button>
      </div>
    ) : null;
  CreateSearchTools.displayName = "CreateSearchTools";
  return { default: CreateSearchTools };
});

vi.mock("@/components/common_components/DeleteResourceModal", () => {
  const DeleteResourceModal = ({
    isOpen,
    onOk,
    onCancel,
  }: {
    isOpen: boolean;
    onOk: () => void;
    onCancel: () => void;
  }) =>
    isOpen ? (
      <div data-testid="delete-resource-modal">
        <button onClick={onOk}>Confirm Delete</button>
        <button onClick={onCancel}>Cancel Delete</button>
      </div>
    ) : null;
  DeleteResourceModal.displayName = "DeleteResourceModal";
  return { default: DeleteResourceModal };
});

const mockSearchTools: SearchTool[] = [
  {
    search_tool_id: "tool-1",
    search_tool_name: "Perplexity Search",
    litellm_params: {
      search_provider: "perplexity",
      api_key: "sk-test-key",
    },
    search_tool_info: {
      description: "Test description",
    },
    created_at: "2024-01-15T10:30:00Z",
  },
  {
    search_tool_id: "tool-2",
    search_tool_name: "Tavily Search",
    litellm_params: {
      search_provider: "tavily",
    },
    created_at: "2024-01-16T10:30:00Z",
  },
];

const mockAvailableProviders: AvailableSearchProvider[] = [
  {
    provider_name: "perplexity",
    ui_friendly_name: "Perplexity AI",
  },
  {
    provider_name: "tavily",
    ui_friendly_name: "Tavily Search",
  },
];

describe("SearchTools", () => {
  const defaultProps = {
    accessToken: "test-token",
    userRole: "Admin",
    userID: "user-1",
  };

  beforeEach(() => {
    vi.clearAllMocks();
    testQueryClient.clear();
    searchToolViewMounts.count = 0;
    vi.mocked(networking.fetchSearchTools).mockResolvedValue({ search_tools: mockSearchTools });
    vi.mocked(networking.fetchAvailableSearchProviders).mockResolvedValue({ providers: mockAvailableProviders });
    vi.mocked(roles.isAdminRole).mockReturnValue(true);
  });

  it("should render", async () => {
    renderWithProviders(<SearchTools {...defaultProps} />);
    await waitFor(() => {
      expect(screen.getByText("Search Tools")).toBeInTheDocument();
    });
  });

  it("should display missing authentication parameters message when accessToken is missing", () => {
    renderWithProviders(<SearchTools {...defaultProps} accessToken={null} />);
    expect(screen.getByText("Missing required authentication parameters.")).toBeInTheDocument();
  });

  it("should display missing authentication parameters message when userRole is missing", () => {
    renderWithProviders(<SearchTools {...defaultProps} userRole={null} />);
    expect(screen.getByText("Missing required authentication parameters.")).toBeInTheDocument();
  });

  it("should display missing authentication parameters message when userID is missing", () => {
    renderWithProviders(<SearchTools {...defaultProps} userID={null} />);
    expect(screen.getByText("Missing required authentication parameters.")).toBeInTheDocument();
  });

  it("should display search tools table with tools", async () => {
    renderWithProviders(<SearchTools {...defaultProps} />);
    await waitFor(() => {
      expect(screen.getByText("Perplexity Search")).toBeInTheDocument();
    });
    expect(screen.getAllByText("Tavily Search").length).toBeGreaterThan(0);
  });

  it("should display empty state when no search tools are available", async () => {
    vi.mocked(networking.fetchSearchTools).mockResolvedValue({ search_tools: [] });

    renderWithProviders(<SearchTools {...defaultProps} />);
    await waitFor(() => {
      expect(screen.getByText("No search tools configured")).toBeInTheDocument();
    });
  });

  it("should show Add New Search Tool button when user is admin", async () => {
    renderWithProviders(<SearchTools {...defaultProps} />);
    await waitFor(() => {
      expect(screen.getByRole("button", { name: /add new search tool/i })).toBeInTheDocument();
    });
  });

  it("should not show Add New Search Tool button when user is not admin", async () => {
    vi.mocked(roles.isAdminRole).mockReturnValue(false);

    renderWithProviders(<SearchTools {...defaultProps} />);
    await waitFor(() => {
      expect(screen.getByText("Search Tools")).toBeInTheDocument();
    });
    expect(screen.queryByRole("button", { name: /add new search tool/i })).not.toBeInTheDocument();
  });

  it("should open create modal when Add New Search Tool button is clicked", async () => {
    const user = userEvent.setup({ delay: null });
    renderWithProviders(<SearchTools {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /add new search tool/i })).toBeInTheDocument();
    });

    const addButton = screen.getByRole("button", { name: /add new search tool/i });
    await user.click(addButton);

    expect(screen.getByTestId("create-search-tool-modal")).toBeInTheDocument();
  });

  it("should navigate to tool view when tool ID is clicked", async () => {
    const user = userEvent.setup({ delay: null });
    renderWithProviders(<SearchTools {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText("Perplexity Search")).toBeInTheDocument();
    });

    const toolIdButton = screen.getByRole("button", { name: /tool-1/i });
    await user.click(toolIdButton);

    await waitFor(() => {
      expect(screen.getByTestId("search-tool-view")).toBeInTheDocument();
    });
    expect(screen.getByText(/Search Tool View: Perplexity Search/i)).toBeInTheDocument();
  });

  it("should navigate back from tool view to table", async () => {
    const user = userEvent.setup({ delay: null });
    renderWithProviders(<SearchTools {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText("Perplexity Search")).toBeInTheDocument();
    });

    const toolIdButton = screen.getByRole("button", { name: /tool-1/i });
    await user.click(toolIdButton);

    await waitFor(() => {
      expect(screen.getByTestId("search-tool-view")).toBeInTheDocument();
    });

    const backButton = screen.getByRole("button", { name: /back/i });
    await user.click(backButton);

    await waitFor(() => {
      expect(screen.queryByTestId("search-tool-view")).not.toBeInTheDocument();
      expect(screen.getByText("Perplexity Search")).toBeInTheDocument();
    });
  });

  describe("URL state", () => {
    const lastUrl = (onUrlUpdate: ReturnType<typeof vi.fn<OnUrlUpdateFunction>>) => onUrlUpdate.mock.calls.at(-1)?.[0];

    const renderWithStableUrl = (searchParams: Record<string, string>, onUrlUpdate: OnUrlUpdateFunction) =>
      render(
        <NuqsTestingAdapter
          searchParams={searchParams}
          onUrlUpdate={onUrlUpdate}
          hasMemory
          resetUrlUpdateQueueOnMount={false}
        >
          <QueryClientProvider client={testQueryClient}>
            <SearchTools {...defaultProps} />
          </QueryClientProvider>
        </NuqsTestingAdapter>,
      );

    it("opens the tool named by search_tool in the URL", async () => {
      renderWithProviders(<SearchTools {...defaultProps} />, { searchParams: { search_tool: "tool-2" } });
      expect(await screen.findByText(/Search Tool View: Tavily Search/)).toBeInTheDocument();
      expect(screen.queryByRole("button", { name: /tool-1/ })).not.toBeInTheDocument();
    });

    it("falls back to the table and clears search_tool when it names an unknown tool", async () => {
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderWithStableUrl({ search_tool: "missing" }, onUrlUpdate);

      expect(await screen.findByRole("button", { name: /tool-1/ })).toBeInTheDocument();
      expect(screen.queryByTestId("search-tool-view")).not.toBeInTheDocument();
      await waitFor(() => expect(lastUrl(onUrlUpdate)?.searchParams.has("search_tool")).toBe(false));
      expect(lastUrl(onUrlUpdate)?.options.history).toBe("replace");
    });

    it("shows a loading state for a search_tool deep link until the tools arrive", async () => {
      let resolveTools: (value: { search_tools: SearchTool[] }) => void = () => {};
      vi.mocked(networking.fetchSearchTools).mockReturnValue(
        new Promise((resolve) => {
          resolveTools = resolve;
        }),
      );
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderWithStableUrl({ search_tool: "tool-2" }, onUrlUpdate);

      expect(await screen.findByText("Loading search tool…")).toBeInTheDocument();
      expect(screen.queryByTestId("search-tool-view")).not.toBeInTheDocument();
      expect(screen.queryByText("No search tools configured")).not.toBeInTheDocument();

      resolveTools({ search_tools: mockSearchTools });

      expect(await screen.findByText(/Search Tool View: Tavily Search/)).toBeInTheDocument();
      expect(onUrlUpdate).not.toHaveBeenCalled();
    });

    it("keeps the open tool view mounted when the page opens a modal", async () => {
      const user = userEvent.setup({ delay: null });
      renderWithProviders(<SearchTools {...defaultProps} />, { searchParams: { search_tool: "tool-2" } });
      await screen.findByText(/Search Tool View: Tavily Search/);
      expect(searchToolViewMounts.count).toBe(1);

      await user.click(screen.getByRole("button", { name: /add new search tool/i }));

      expect(screen.getByTestId("create-search-tool-modal")).toBeInTheDocument();
      expect(screen.getByTestId("search-tool-view")).toBeInTheDocument();
      expect(searchToolViewMounts.count).toBe(1);
    });

    it("pushes search_tool when a tool ID is clicked and clears it on Back", async () => {
      const user = userEvent.setup({ delay: null });
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderWithProviders(<SearchTools {...defaultProps} />, { onUrlUpdate });

      await user.click(await screen.findByRole("button", { name: /tool-1/ }));
      await waitFor(() => expect(lastUrl(onUrlUpdate)?.searchParams.get("search_tool")).toBe("tool-1"));
      expect(lastUrl(onUrlUpdate)?.options.history).toBe("push");

      await user.click(await screen.findByRole("button", { name: "Back" }));
      await waitFor(() => expect(lastUrl(onUrlUpdate)?.searchParams.has("search_tool")).toBe(false));
      expect(await screen.findByRole("button", { name: /tool-1/ })).toBeInTheDocument();
    });

    it("keeps the edit modal out of the URL and on the table", async () => {
      const user = userEvent.setup({ delay: null });
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderWithProviders(<SearchTools {...defaultProps} />, { onUrlUpdate });

      await screen.findByText("Perplexity Search");
      await user.click(screen.getByTestId("search-tool-actions-tool-1"));
      await user.click(await screen.findByTestId("search-tool-action-edit"));
      await waitFor(() => expect(screen.getByLabelText("Search Tool Name")).toHaveValue("Perplexity Search"));
      expect(screen.queryByTestId("search-tool-view")).not.toBeInTheDocument();

      await user.click(screen.getByRole("button", { name: "Cancel" }));
      await waitFor(() => expect(screen.queryByLabelText("Search Tool Name")).not.toBeInTheDocument());
      expect(screen.queryByTestId("search-tool-view")).not.toBeInTheDocument();
      expect(onUrlUpdate).not.toHaveBeenCalled();
    });
  });
});
