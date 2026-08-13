import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { vectorStoreSearchCall } from "@/components/networking";

import { VectorStoreTester } from "./VectorStoreTester";

vi.mock("@/components/networking", () => ({
  vectorStoreSearchCall: vi.fn(),
}));

const mockWarning = vi.fn();
vi.mock("@/components/molecules/message_manager", () => ({
  __esModule: true,
  default: { warning: (...args: unknown[]) => mockWarning(...args) },
}));

const mockFromBackend = vi.fn();
const mockSuccess = vi.fn();
vi.mock("@/components/molecules/notifications_manager", () => ({
  __esModule: true,
  default: {
    fromBackend: (...args: unknown[]) => mockFromBackend(...args),
    success: (...args: unknown[]) => mockSuccess(...args),
  },
}));

const mockSearch = vi.mocked(vectorStoreSearchCall);

const searchResponse = {
  object: "vector_store.search_results.page",
  search_query: "hello",
  data: [
    {
      score: 0.91234,
      content: [{ text: "the quick brown fox", type: "text" }],
      file_id: "file-1",
      filename: "notes.txt",
      attributes: { source: "manual" },
    },
  ],
};

const EMPTY_STATE = "Test your vector store by entering a search query below";

const renderTester = () => render(<VectorStoreTester vectorStoreId="vs_123" accessToken="sk-test" />);

const queryInput = () => screen.getByPlaceholderText(/enter your search query/i);
const searchButton = () => screen.getByRole("button", { name: /search/i });

describe("VectorStoreTester", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockSearch.mockResolvedValue(searchResponse);
  });

  it("shows the empty state before any search has run", () => {
    renderTester();
    expect(screen.getByText(EMPTY_STATE)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /clear history/i })).not.toBeInTheDocument();
  });

  it("does not search until a non-blank query is entered", async () => {
    const user = userEvent.setup();
    renderTester();

    await user.click(searchButton());
    expect(mockSearch).not.toHaveBeenCalled();

    await user.type(queryInput(), "hello");
    await user.click(searchButton());

    await waitFor(() => expect(mockSearch).toHaveBeenCalledWith("sk-test", "vs_123", "hello"));
  });

  it("renders the returned result and clears the query input", async () => {
    const user = userEvent.setup();
    renderTester();

    await user.type(queryInput(), "hello");
    await user.click(searchButton());

    expect(await screen.findByText("Result 1")).toBeInTheDocument();
    expect(screen.getByText("1 results")).toBeInTheDocument();
    expect(screen.getByText("Score: 0.9123")).toBeInTheDocument();
    expect(screen.queryByText(EMPTY_STATE)).not.toBeInTheDocument();
    await waitFor(() => expect(queryInput()).toHaveValue(""));
  });

  it("expands a result to reveal its content and metadata", async () => {
    const user = userEvent.setup();
    renderTester();

    await user.type(queryInput(), "hello");
    await user.click(searchButton());

    expect(await screen.findByText("Result 1")).toBeInTheDocument();
    expect(screen.queryByText("the quick brown fox")).not.toBeInTheDocument();

    await user.click(screen.getByText("Result 1"));

    expect(screen.getByText("the quick brown fox")).toBeInTheDocument();
    expect(screen.getByText("File ID:").parentElement).toHaveTextContent("file-1");
    expect(screen.getByText("Filename:").parentElement).toHaveTextContent("notes.txt");
  });

  it("warns instead of searching when the query is only whitespace", async () => {
    const user = userEvent.setup();
    renderTester();

    await user.type(queryInput(), "   ");
    await user.type(queryInput(), "{Enter}");

    expect(mockWarning).toHaveBeenCalledWith("Please enter a search query");
    expect(mockSearch).not.toHaveBeenCalled();
  });

  it("submits on Enter but not on Shift+Enter", async () => {
    const user = userEvent.setup();
    renderTester();

    await user.type(queryInput(), "hello");
    await user.type(queryInput(), "{Shift>}{Enter}{/Shift}");
    expect(mockSearch).not.toHaveBeenCalled();

    await user.type(queryInput(), "{Enter}");
    await waitFor(() => expect(mockSearch).toHaveBeenCalledTimes(1));
  });

  it("reports a failed search and keeps the history empty", async () => {
    const user = userEvent.setup();
    mockSearch.mockRejectedValue(new Error("boom"));
    renderTester();

    await user.type(queryInput(), "hello");
    await user.click(searchButton());

    await waitFor(() => expect(mockFromBackend).toHaveBeenCalledWith("Failed to search vector store"));
    expect(screen.getByText(EMPTY_STATE)).toBeInTheDocument();
  });

  it("clears the search history", async () => {
    const user = userEvent.setup();
    renderTester();

    await user.type(queryInput(), "hello");
    await user.click(searchButton());
    expect(await screen.findByText("Result 1")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /clear history/i }));

    expect(screen.queryByText("Result 1")).not.toBeInTheDocument();
    expect(screen.getByText(EMPTY_STATE)).toBeInTheDocument();
  });
});
