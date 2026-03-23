import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { SearchToolView } from "./SearchToolView";
import { AvailableSearchProvider, SearchTool } from "./types";

vi.mock("@/utils/dataUtils", () => ({
  copyToClipboard: vi.fn().mockResolvedValue(true),
}));

vi.mock("./SearchToolTester", () => ({
  SearchToolTester: ({ searchToolName, accessToken }: { searchToolName: string; accessToken: string }) => (
    <div data-testid="search-tool-tester">
      <span>Search Tool Tester for {searchToolName}</span>
      <span>Access Token: {accessToken}</span>
    </div>
  ),
}));

describe("SearchToolView", () => {
  const mockSearchTool: SearchTool = {
    search_tool_id: "test-tool-id-123",
    search_tool_name: "Test Search Tool",
    litellm_params: {
      search_provider: "perplexity",
      api_key: "sk-test-key",
    },
    search_tool_info: {
      description: "Test description",
    },
    created_at: "2024-01-15T10:30:00Z",
  };

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

  const defaultProps = {
    searchTool: mockSearchTool,
    onBack: vi.fn(),
    isEditing: false,
    accessToken: "test-token",
    availableProviders: mockAvailableProviders,
  };

  beforeEach(async () => {
    vi.clearAllMocks();
    const { copyToClipboard } = await import("@/utils/dataUtils");
    vi.mocked(copyToClipboard).mockResolvedValue(true);
  });

  it("should render", () => {
    render(<SearchToolView {...defaultProps} />);
    expect(screen.getByText("Test Search Tool")).toBeInTheDocument();
  });

  it("should display search tool name", () => {
    render(<SearchToolView {...defaultProps} />);
    expect(screen.getByText("Test Search Tool")).toBeInTheDocument();
  });

  it("should display search tool ID", () => {
    render(<SearchToolView {...defaultProps} />);
    expect(screen.getByText("test-tool-id-123")).toBeInTheDocument();
  });

  it("should display provider name using UI-friendly name when available", () => {
    render(<SearchToolView {...defaultProps} />);
    expect(screen.getByText("Perplexity AI")).toBeInTheDocument();
  });

  it("should display provider name using provider_name when UI-friendly name is not available", () => {
    const searchToolWithoutProvider: SearchTool = {
      ...mockSearchTool,
      litellm_params: {
        search_provider: "unknown-provider",
      },
    };

    render(
      <SearchToolView
        {...defaultProps}
        searchTool={searchToolWithoutProvider}
      />,
    );
    expect(screen.getByText("unknown-provider")).toBeInTheDocument();
  });

  it("should display masked API key when API key is set", () => {
    render(<SearchToolView {...defaultProps} />);
    expect(screen.getByText("****")).toBeInTheDocument();
  });

  it("should display 'Not set' when API key is not set", () => {
    const searchToolWithoutApiKey: SearchTool = {
      ...mockSearchTool,
      litellm_params: {
        search_provider: "perplexity",
      },
    };

    render(
      <SearchToolView
        {...defaultProps}
        searchTool={searchToolWithoutApiKey}
      />,
    );
    expect(screen.getByText("Not set")).toBeInTheDocument();
  });

  it("should display formatted created_at date", () => {
    render(<SearchToolView {...defaultProps} />);
    const dateText = screen.getByText(/2024-01-15/);
    expect(dateText).toBeInTheDocument();
  });

  it("should display 'Unknown' when created_at is not set", () => {
    const searchToolWithoutDate: SearchTool = {
      ...mockSearchTool,
      created_at: undefined,
    };

    render(
      <SearchToolView
        {...defaultProps}
        searchTool={searchToolWithoutDate}
      />,
    );
    expect(screen.getByText("Unknown")).toBeInTheDocument();
  });

  it("should display description when search_tool_info.description is provided", () => {
    render(<SearchToolView {...defaultProps} />);
    expect(screen.getByText("Test description")).toBeInTheDocument();
  });

  it("should not display description card when search_tool_info.description is not provided", () => {
    const searchToolWithoutDescription: SearchTool = {
      ...mockSearchTool,
      search_tool_info: {},
    };

    render(
      <SearchToolView
        {...defaultProps}
        searchTool={searchToolWithoutDescription}
      />,
    );
    expect(screen.queryByText("Description")).not.toBeInTheDocument();
  });

  it("should call onBack when back button is clicked", async () => {
    const user = userEvent.setup({ delay: null });
    const onBack = vi.fn();
    render(<SearchToolView {...defaultProps} onBack={onBack} />);

    const backButton = screen.getByRole("button", { name: /back to all search tools/i });
    await user.click(backButton);

    expect(onBack).toHaveBeenCalledTimes(1);
  });

  it("should copy search tool name to clipboard when copy button is clicked", async () => {
    const user = userEvent.setup({ delay: null });
    const { copyToClipboard } = await import("@/utils/dataUtils");
    render(<SearchToolView {...defaultProps} />);

    const toolNameContainer = screen.getByText("Test Search Tool").closest("div");
    expect(toolNameContainer).toBeInTheDocument();

    const copyButtons = within(toolNameContainer!).getAllByRole("button");
    const nameCopyButton = copyButtons.find((button) => {
      return button.querySelector("svg") !== null;
    });

    expect(nameCopyButton).toBeInTheDocument();
    await user.click(nameCopyButton!);

    await waitFor(() => {
      expect(copyToClipboard).toHaveBeenCalledWith("Test Search Tool");
    });
  });

  it("should copy search tool ID to clipboard when copy button is clicked", async () => {
    const user = userEvent.setup({ delay: null });
    const { copyToClipboard } = await import("@/utils/dataUtils");
    render(<SearchToolView {...defaultProps} />);

    const toolIdContainer = screen.getByText("test-tool-id-123").closest("div");
    expect(toolIdContainer).toBeInTheDocument();

    const copyButtons = within(toolIdContainer!).getAllByRole("button");
    const idCopyButton = copyButtons.find((button) => {
      return button.querySelector("svg") !== null;
    });

    expect(idCopyButton).toBeInTheDocument();
    await user.click(idCopyButton!);

    await waitFor(() => {
      expect(copyToClipboard).toHaveBeenCalledWith("test-tool-id-123");
    });
  });

  it("should show check icon after copying search tool name", async () => {
    const user = userEvent.setup({ delay: null });
    const { copyToClipboard } = await import("@/utils/dataUtils");
    vi.mocked(copyToClipboard).mockResolvedValue(true);

    render(<SearchToolView {...defaultProps} />);

    const toolNameContainer = screen.getByText("Test Search Tool").closest("div");
    const copyButtons = within(toolNameContainer!).getAllByRole("button");
    const nameCopyButton = copyButtons.find((button) => {
      return button.querySelector("svg") !== null;
    });

    expect(nameCopyButton).toBeInTheDocument();

    const initialSvg = nameCopyButton!.querySelector("svg");
    expect(initialSvg).toBeInTheDocument();

    await user.click(nameCopyButton!);

    await waitFor(() => {
      const updatedSvg = nameCopyButton!.querySelector("svg");
      expect(updatedSvg).toBeInTheDocument();
      expect(nameCopyButton).toHaveClass("text-green-600");
    });
  });


  it("should not show check icon when copy fails", async () => {
    const user = userEvent.setup({ delay: null });
    const { copyToClipboard } = await import("@/utils/dataUtils");
    vi.mocked(copyToClipboard).mockResolvedValue(false);

    render(<SearchToolView {...defaultProps} />);

    const toolNameContainer = screen.getByText("Test Search Tool").closest("div");
    const copyButtons = within(toolNameContainer!).getAllByRole("button");
    const nameCopyButton = copyButtons.find((button) => {
      return button.querySelector("svg") !== null;
    });

    expect(nameCopyButton).toBeInTheDocument();
    await user.click(nameCopyButton!);

    await waitFor(() => {
      expect(copyToClipboard).toHaveBeenCalledWith("Test Search Tool");
    }, { timeout: 3000 });

    expect(nameCopyButton).not.toHaveClass("text-green-600");
  });

  it("should render SearchToolTester when accessToken is provided", () => {
    render(<SearchToolView {...defaultProps} />);
    expect(screen.getByTestId("search-tool-tester")).toBeInTheDocument();
    expect(screen.getByText(/Search Tool Tester for Test Search Tool/)).toBeInTheDocument();
  });

  it("should not render SearchToolTester when accessToken is null", () => {
    render(<SearchToolView {...defaultProps} accessToken={null} />);
    expect(screen.queryByTestId("search-tool-tester")).not.toBeInTheDocument();
  });

  it("should pass correct props to SearchToolTester", () => {
    render(<SearchToolView {...defaultProps} />);
    expect(screen.getByText("Access Token: test-token")).toBeInTheDocument();
  });
});
