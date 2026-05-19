import * as roles from "@/utils/roles";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import * as networking from "../networking";
import SearchTools from "./SearchTools";
import { AvailableSearchProvider, SearchTool } from "./types";

vi.mock("../networking", () => ({
  fetchSearchTools: vi.fn(),
  updateSearchTool: vi.fn(),
  deleteSearchTool: vi.fn(),
  fetchAvailableSearchProviders: vi.fn(),
}));

vi.mock("@/utils/roles", () => ({
  isAdminRole: vi.fn(),
}));

vi.mock("./SearchToolView", () => {
  const SearchToolView = ({ searchTool, onBack }: { searchTool: SearchTool; onBack: () => void }) => (
    <div data-testid="search-tool-view">
      <div>Search Tool View: {searchTool.search_tool_name}</div>
      <button onClick={onBack}>Back</button>
    </div>
  );
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

vi.mock("../common_components/DeleteResourceModal", () => {
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

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });
  const Wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
  Wrapper.displayName = "TestWrapper";
  return Wrapper;
};

describe("SearchTools", () => {
  const defaultProps = {
    accessToken: "test-token",
    userRole: "Admin",
    userID: "user-1",
  };

  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(networking.fetchSearchTools).mockResolvedValue({ search_tools: mockSearchTools });
    vi.mocked(networking.fetchAvailableSearchProviders).mockResolvedValue({ providers: mockAvailableProviders });
    vi.mocked(roles.isAdminRole).mockReturnValue(true);
  });

  it("should render", async () => {
    render(<SearchTools {...defaultProps} />, { wrapper: createWrapper() });
    await waitFor(() => {
      expect(screen.getByText("Search Tools")).toBeInTheDocument();
    });
  });

  it("should display missing authentication parameters message when accessToken is missing", () => {
    render(<SearchTools {...defaultProps} accessToken={null} />, { wrapper: createWrapper() });
    expect(screen.getByText("Missing required authentication parameters.")).toBeInTheDocument();
  });

  it("should display missing authentication parameters message when userRole is missing", () => {
    render(<SearchTools {...defaultProps} userRole={null} />, { wrapper: createWrapper() });
    expect(screen.getByText("Missing required authentication parameters.")).toBeInTheDocument();
  });

  it("should display missing authentication parameters message when userID is missing", () => {
    render(<SearchTools {...defaultProps} userID={null} />, { wrapper: createWrapper() });
    expect(screen.getByText("Missing required authentication parameters.")).toBeInTheDocument();
  });

  it("should display search tools table with tools", async () => {
    render(<SearchTools {...defaultProps} />, { wrapper: createWrapper() });
    await waitFor(() => {
      expect(screen.getByText("Perplexity Search")).toBeInTheDocument();
    });
    expect(screen.getAllByText("Tavily Search").length).toBeGreaterThan(0);
  });

  it("should display empty state when no search tools are available", async () => {
    vi.mocked(networking.fetchSearchTools).mockResolvedValue({ search_tools: [] });

    render(<SearchTools {...defaultProps} />, { wrapper: createWrapper() });
    await waitFor(() => {
      expect(screen.getByText("No search tools configured")).toBeInTheDocument();
    });
  });

  it("should show Add New Search Tool button when user is admin", async () => {
    render(<SearchTools {...defaultProps} />, { wrapper: createWrapper() });
    await waitFor(() => {
      expect(screen.getByRole("button", { name: /add new search tool/i })).toBeInTheDocument();
    });
  });

  it("should not show Add New Search Tool button when user is not admin", async () => {
    vi.mocked(roles.isAdminRole).mockReturnValue(false);

    render(<SearchTools {...defaultProps} />, { wrapper: createWrapper() });
    await waitFor(() => {
      expect(screen.getByText("Search Tools")).toBeInTheDocument();
    });
    expect(screen.queryByRole("button", { name: /add new search tool/i })).not.toBeInTheDocument();
  });

  it("should open create modal when Add New Search Tool button is clicked", async () => {
    const user = userEvent.setup({ delay: null });
    render(<SearchTools {...defaultProps} />, { wrapper: createWrapper() });

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /add new search tool/i })).toBeInTheDocument();
    });

    const addButton = screen.getByRole("button", { name: /add new search tool/i });
    await user.click(addButton);

    expect(screen.getByTestId("create-search-tool-modal")).toBeInTheDocument();
  });

  it("should navigate to tool view when tool ID is clicked", async () => {
    const user = userEvent.setup({ delay: null });
    render(<SearchTools {...defaultProps} />, { wrapper: createWrapper() });

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
    render(<SearchTools {...defaultProps} />, { wrapper: createWrapper() });

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
});
