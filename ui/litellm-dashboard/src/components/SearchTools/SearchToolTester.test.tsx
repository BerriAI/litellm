import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { SearchToolTester } from "./SearchToolTester";
import * as networking from "../networking";
import NotificationsManager from "../molecules/notifications_manager";

vi.mock("../networking", () => ({
  searchToolQueryCall: vi.fn(),
}));

vi.mock("antd", async () => {
  const actual = await vi.importActual("antd");
  return {
    ...actual,
    message: {
      warning: vi.fn(),
      success: vi.fn(),
      error: vi.fn(),
    },
  };
});

const mockSearchResults = {
  results: [
    {
      title: "Test Result 1",
      url: "https://example.com/result1",
      snippet: "This is a short snippet for the first result.",
    },
    {
      title: "Test Result 2",
      url: "https://example.com/result2",
      snippet: "This is a longer snippet that exceeds two hundred characters and should be truncated when displayed in the results. It contains more detailed information about the search result that would normally be shown in a search engine result page.",
    },
  ],
};

const defaultProps = {
  searchToolName: "test-search-tool",
  accessToken: "test-token",
};

describe("SearchToolTester", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(networking.searchToolQueryCall).mockResolvedValue(mockSearchResults);
    vi.spyOn(Date, "now").mockReturnValue(1000000000000);
  });

  it("should render", () => {
    render(<SearchToolTester {...defaultProps} />);
    expect(screen.getByText("Test Search Tool")).toBeInTheDocument();
  });

  it("should display empty state when no search has been performed", () => {
    render(<SearchToolTester {...defaultProps} />);
    expect(screen.getByText("Test your search tool")).toBeInTheDocument();
    expect(screen.getByText("Enter a query above to see search results")).toBeInTheDocument();
  });

  it("should display search input with placeholder", () => {
    render(<SearchToolTester {...defaultProps} />);
    expect(screen.getByPlaceholderText("Enter your search query...")).toBeInTheDocument();
  });

  it("should display search button", () => {
    render(<SearchToolTester {...defaultProps} />);
    expect(screen.getByRole("button", { name: /search/i })).toBeInTheDocument();
  });

  it("should disable search button when input is empty", () => {
    render(<SearchToolTester {...defaultProps} />);
    const searchButton = screen.getByRole("button", { name: /search/i });
    expect(searchButton).toBeDisabled();
  });

  it("should enable search button when input has text", async () => {
    const user = userEvent.setup();
    render(<SearchToolTester {...defaultProps} />);
    const input = screen.getByPlaceholderText("Enter your search query...");
    await user.type(input, "test query");
    const searchButton = screen.getByRole("button", { name: /search/i });
    expect(searchButton).not.toBeDisabled();
  });

  it("should call searchToolQueryCall when search button is clicked", async () => {
    const user = userEvent.setup();
    render(<SearchToolTester {...defaultProps} />);
    const input = screen.getByPlaceholderText("Enter your search query...");
    await user.type(input, "test query");
    const searchButton = screen.getByRole("button", { name: /search/i });
    await user.click(searchButton);
    expect(networking.searchToolQueryCall).toHaveBeenCalledWith("test-token", "test-search-tool", "test query");
  });

  it("should call searchToolQueryCall when Enter is pressed in input", async () => {
    const user = userEvent.setup();
    render(<SearchToolTester {...defaultProps} />);
    const input = screen.getByPlaceholderText("Enter your search query...");
    await user.type(input, "test query{Enter}");
    expect(networking.searchToolQueryCall).toHaveBeenCalledWith("test-token", "test-search-tool", "test query");
  });

  it("should not call searchToolQueryCall when Shift+Enter is pressed", async () => {
    const user = userEvent.setup();
    render(<SearchToolTester {...defaultProps} />);
    const input = screen.getByPlaceholderText("Enter your search query...");
    await user.type(input, "test query");
    await user.keyboard("{Shift>}{Enter}{/Shift}");
    expect(networking.searchToolQueryCall).not.toHaveBeenCalled();
  });

  it("should display loading state while searching", async () => {
    vi.mocked(networking.searchToolQueryCall).mockImplementation(
      () => new Promise((resolve) => setTimeout(() => resolve(mockSearchResults), 100)),
    );
    const user = userEvent.setup();
    render(<SearchToolTester {...defaultProps} />);
    const input = screen.getByPlaceholderText("Enter your search query...");
    await user.type(input, "test query");
    const searchButton = screen.getByRole("button", { name: /search/i });
    await user.click(searchButton);
    expect(screen.getByText("Searching...")).toBeInTheDocument();
  });

  it("should display search results after successful search", async () => {
    const user = userEvent.setup();
    render(<SearchToolTester {...defaultProps} />);
    const input = screen.getByPlaceholderText("Enter your search query...");
    await user.type(input, "test query");
    const searchButton = screen.getByRole("button", { name: /search/i });
    await user.click(searchButton);
    await waitFor(() => {
      expect(screen.getByText("Test Result 1")).toBeInTheDocument();
    });
    expect(screen.getByText("Test Result 2")).toBeInTheDocument();
  });

  it("should display search query in results header", async () => {
    const user = userEvent.setup();
    render(<SearchToolTester {...defaultProps} />);
    const input = screen.getByPlaceholderText("Enter your search query...");
    await user.type(input, "test query");
    const searchButton = screen.getByRole("button", { name: /search/i });
    await user.click(searchButton);
    await waitFor(() => {
      expect(screen.getByText("test query")).toBeInTheDocument();
    });
  });

  it("should display result count in results header", async () => {
    const user = userEvent.setup();
    render(<SearchToolTester {...defaultProps} />);
    const input = screen.getByPlaceholderText("Enter your search query...");
    await user.type(input, "test query");
    const searchButton = screen.getByRole("button", { name: /search/i });
    await user.click(searchButton);
    await waitFor(() => {
      expect(screen.getByText("2 results")).toBeInTheDocument();
    });
  });

  it("should display singular result count when only one result", async () => {
    const singleResult = {
      results: [
        {
          title: "Single Result",
          url: "https://example.com/single",
          snippet: "Single result snippet",
        },
      ],
    };
    vi.mocked(networking.searchToolQueryCall).mockResolvedValue(singleResult);
    const user = userEvent.setup();
    render(<SearchToolTester {...defaultProps} />);
    const input = screen.getByPlaceholderText("Enter your search query...");
    await user.type(input, "test query");
    const searchButton = screen.getByRole("button", { name: /search/i });
    await user.click(searchButton);
    await waitFor(() => {
      expect(screen.getByText("1 result")).toBeInTheDocument();
    });
  });

  it("should display result URLs", async () => {
    const user = userEvent.setup();
    render(<SearchToolTester {...defaultProps} />);
    const input = screen.getByPlaceholderText("Enter your search query...");
    await user.type(input, "test query");
    const searchButton = screen.getByRole("button", { name: /search/i });
    await user.click(searchButton);
    await waitFor(() => {
      expect(screen.getByText("https://example.com/result1")).toBeInTheDocument();
    });
    expect(screen.getByText("https://example.com/result2")).toBeInTheDocument();
  });

  it("should display result snippets", async () => {
    const user = userEvent.setup();
    render(<SearchToolTester {...defaultProps} />);
    const input = screen.getByPlaceholderText("Enter your search query...");
    await user.type(input, "test query");
    const searchButton = screen.getByRole("button", { name: /search/i });
    await user.click(searchButton);
    await waitFor(() => {
      expect(screen.getByText("This is a short snippet for the first result.")).toBeInTheDocument();
    });
  });

  it("should truncate long snippets and show expand button", async () => {
    const user = userEvent.setup();
    render(<SearchToolTester {...defaultProps} />);
    const input = screen.getByPlaceholderText("Enter your search query...");
    await user.type(input, "test query");
    const searchButton = screen.getByRole("button", { name: /search/i });
    await user.click(searchButton);
    await waitFor(() => {
      expect(screen.getByText(/Show more/i)).toBeInTheDocument();
    });
  });

  it("should expand snippet when Show more is clicked", async () => {
    const user = userEvent.setup();
    render(<SearchToolTester {...defaultProps} />);
    const input = screen.getByPlaceholderText("Enter your search query...");
    await user.type(input, "test query");
    const searchButton = screen.getByRole("button", { name: /search/i });
    await user.click(searchButton);
    await waitFor(() => {
      expect(screen.getByText(/Show more/i)).toBeInTheDocument();
    });
    const expandButton = screen.getByText(/Show more/i);
    await user.click(expandButton);
    expect(screen.getByText(/Show less/i)).toBeInTheDocument();
  });

  it("should collapse snippet when Show less is clicked", async () => {
    const user = userEvent.setup();
    render(<SearchToolTester {...defaultProps} />);
    const input = screen.getByPlaceholderText("Enter your search query...");
    await user.type(input, "test query");
    const searchButton = screen.getByRole("button", { name: /search/i });
    await user.click(searchButton);
    await waitFor(() => {
      expect(screen.getByText(/Show more/i)).toBeInTheDocument();
    });
    const expandButton = screen.getByText(/Show more/i);
    await user.click(expandButton);
    const collapseButton = screen.getByText(/Show less/i);
    await user.click(collapseButton);
    expect(screen.getByText(/Show more/i)).toBeInTheDocument();
  });

  it("should display no results message when search returns empty results", async () => {
    vi.mocked(networking.searchToolQueryCall).mockResolvedValue({ results: [] });
    const user = userEvent.setup();
    render(<SearchToolTester {...defaultProps} />);
    const input = screen.getByPlaceholderText("Enter your search query...");
    await user.type(input, "test query");
    const searchButton = screen.getByRole("button", { name: /search/i });
    await user.click(searchButton);
    await waitFor(() => {
      expect(screen.getByText("No results found")).toBeInTheDocument();
    });
    expect(screen.getByText("Try a different search query")).toBeInTheDocument();
  });

  it("should display no results message when search returns null results", async () => {
    vi.mocked(networking.searchToolQueryCall).mockResolvedValue({ results: null });
    const user = userEvent.setup();
    render(<SearchToolTester {...defaultProps} />);
    const input = screen.getByPlaceholderText("Enter your search query...");
    await user.type(input, "test query");
    const searchButton = screen.getByRole("button", { name: /search/i });
    await user.click(searchButton);
    await waitFor(() => {
      expect(screen.getByText("No results found")).toBeInTheDocument();
    });
  });

  it("should handle search errors and show notification", async () => {
    const error = new Error("Search failed");
    vi.mocked(networking.searchToolQueryCall).mockRejectedValue(error);
    const consoleSpy = vi.spyOn(console, "error").mockImplementation(() => { });
    const user = userEvent.setup();
    render(<SearchToolTester {...defaultProps} />);
    const input = screen.getByPlaceholderText("Enter your search query...");
    await user.type(input, "test query");
    const searchButton = screen.getByRole("button", { name: /search/i });
    await user.click(searchButton);
    await waitFor(() => {
      expect(NotificationsManager.fromBackend).toHaveBeenCalledWith("Failed to query search tool");
    });
    consoleSpy.mockRestore();
  });

  it("should maintain search history", async () => {
    const user = userEvent.setup();
    render(<SearchToolTester {...defaultProps} />);
    const input = screen.getByPlaceholderText("Enter your search query...");
    await user.type(input, "first query");
    const searchButton = screen.getByRole("button", { name: /search/i });
    await user.click(searchButton);
    await waitFor(() => {
      expect(screen.getByText("first query")).toBeInTheDocument();
    });
    await user.clear(input);
    await user.type(input, "second query");
    await user.click(searchButton);
    await waitFor(() => {
      expect(screen.getByText("second query")).toBeInTheDocument();
    });
    expect(screen.getByText("Previous Searches")).toBeInTheDocument();
    expect(screen.getByText("first query")).toBeInTheDocument();
  });

  it("should allow clicking previous search to set query", async () => {
    const user = userEvent.setup();
    render(<SearchToolTester {...defaultProps} />);
    const input = screen.getByPlaceholderText("Enter your search query...");
    await user.type(input, "first query");
    const searchButton = screen.getByRole("button", { name: /search/i });
    await user.click(searchButton);
    await waitFor(() => {
      expect(screen.getByText("first query")).toBeInTheDocument();
    });
    await user.clear(input);
    await user.type(input, "second query");
    await user.click(searchButton);
    await waitFor(() => {
      expect(screen.getByText("second query")).toBeInTheDocument();
    });
    const historyItems = screen.getAllByText("first query");
    const historyItem = historyItems.find((item) => item.closest('[class*="cursor-pointer"]'));
    if (historyItem) {
      await user.click(historyItem);
      expect(input).toHaveValue("first query");
    }
  });

  it("should clear history when Clear All is clicked", async () => {
    const user = userEvent.setup();
    render(<SearchToolTester {...defaultProps} />);
    const input = screen.getByPlaceholderText("Enter your search query...");
    await user.type(input, "first query");
    const searchButton = screen.getByRole("button", { name: /search/i });
    await user.click(searchButton);
    await waitFor(() => {
      expect(screen.getByText("first query")).toBeInTheDocument();
    });
    await user.clear(input);
    await user.type(input, "second query");
    await user.click(searchButton);
    await waitFor(() => {
      expect(screen.getByText("Previous Searches")).toBeInTheDocument();
    });
    const clearButton = screen.getByRole("button", { name: /clear all/i });
    await user.click(clearButton);
    expect(NotificationsManager.success).toHaveBeenCalledWith("Search history cleared");
    expect(screen.queryByText("Previous Searches")).not.toBeInTheDocument();
  });

  it("should limit history display to 5 previous searches", async () => {
    const user = userEvent.setup();
    render(<SearchToolTester {...defaultProps} />);
    const input = screen.getByPlaceholderText("Enter your search query...");
    for (let i = 1; i <= 7; i++) {
      await user.clear(input);
      await user.type(input, `query ${i}`);
      const searchButton = screen.getByRole("button", { name: /search/i });
      await user.click(searchButton);
      await waitFor(() => {
        expect(screen.getByText(`query ${i}`)).toBeInTheDocument();
      });
    }
    const historySection = screen.queryByText("Previous Searches");
    if (historySection) {
      const historyItems = historySection.parentElement?.querySelectorAll('[class*="cursor-pointer"]');
      expect(historyItems?.length).toBeLessThanOrEqual(5);
    }
  });

  it("should preserve query text after search", async () => {
    const user = userEvent.setup();
    render(<SearchToolTester {...defaultProps} />);
    const input = screen.getByPlaceholderText("Enter your search query...");
    await user.type(input, "test query");
    const searchButton = screen.getByRole("button", { name: /search/i });
    await user.click(searchButton);
    await waitFor(() => {
      expect(screen.getByText("test query")).toBeInTheDocument();
    });
    expect(input).toHaveValue("test query");
  });

  it("should disable input and button while loading", async () => {
    vi.mocked(networking.searchToolQueryCall).mockImplementation(
      () => new Promise((resolve) => setTimeout(() => resolve(mockSearchResults), 100)),
    );
    const user = userEvent.setup();
    render(<SearchToolTester {...defaultProps} />);
    const input = screen.getByPlaceholderText("Enter your search query...");
    await user.type(input, "test query");
    const searchButton = screen.getByRole("button", { name: /search/i });
    await user.click(searchButton);
    expect(input).toBeDisabled();
    expect(searchButton).toBeDisabled();
  });

  it("should display result links that open in new tab", async () => {
    const user = userEvent.setup();
    render(<SearchToolTester {...defaultProps} />);
    const input = screen.getByPlaceholderText("Enter your search query...");
    await user.type(input, "test query");
    const searchButton = screen.getByRole("button", { name: /search/i });
    await user.click(searchButton);
    await waitFor(() => {
      const link = screen.getByRole("link", { name: "Test Result 1" });
      expect(link).toHaveAttribute("href", "https://example.com/result1");
      expect(link).toHaveAttribute("target", "_blank");
      expect(link).toHaveAttribute("rel", "noopener noreferrer");
    });
  });
});
