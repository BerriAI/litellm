import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import * as networking from "@/components/networking";
import * as roles from "@/utils/roles";
import SearchTools from "./SearchTools";
import { SearchTool } from "./types";

vi.mock("@/components/networking", () => ({
  fetchSearchTools: vi.fn(),
  updateSearchTool: vi.fn(),
  deleteSearchTool: vi.fn(),
  fetchAvailableSearchProviders: vi.fn(),
}));

vi.mock("@/utils/roles", () => ({ isAdminRole: vi.fn() }));

vi.mock("@/lib/toast", () => ({
  toast: { success: vi.fn(), error: vi.fn() },
}));

vi.mock("./SearchToolView", () => ({ SearchToolView: () => <div data-testid="search-tool-view" /> }));
vi.mock("./CreateSearchTools", () => ({ default: () => null }));
vi.mock("@/components/common_components/DeleteResourceModal", () => ({ default: () => null }));

const toolWithServerOnlyParams: SearchTool = {
  search_tool_id: "tool-1",
  search_tool_name: "Perplexity Search",
  litellm_params: {
    search_provider: "perplexity",
    api_key: "sk-test-key",
    api_base: "https://api.example.com",
    timeout: 30,
    max_retries: 2,
  },
  search_tool_info: { description: "Test description" },
  created_at: "2024-01-15T10:30:00Z",
};

const toolWithNullServerFields: SearchTool = {
  search_tool_id: "tool-1",
  search_tool_name: "Perplexity Search",
  litellm_params: {
    search_provider: "perplexity",
    api_key: null,
  },
  search_tool_info: { description: null },
  created_at: "2024-01-15T10:30:00Z",
};

const providers = [
  { provider_name: "perplexity", ui_friendly_name: "Perplexity AI" },
  { provider_name: "tavily", ui_friendly_name: "Tavily Search" },
];

const renderPage = () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  return render(
    <QueryClientProvider client={queryClient}>
      <SearchTools accessToken="test-token" userRole="Admin" userID="user-1" />
    </QueryClientProvider>,
  );
};

const openEditModal = async (user: ReturnType<typeof userEvent.setup>) => {
  await screen.findByText("Perplexity Search");
  await user.click(screen.getByTestId("search-tool-actions-tool-1"));
  await user.click(await screen.findByTestId("search-tool-action-edit"));
  await waitFor(() => expect(screen.getByLabelText("Search Tool Name")).toHaveValue("Perplexity Search"));
};

describe("SearchTools edit payload", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(networking.fetchSearchTools).mockResolvedValue({ search_tools: [toolWithServerOnlyParams] });
    vi.mocked(networking.fetchAvailableSearchProviders).mockResolvedValue({ providers });
    vi.mocked(networking.updateSearchTool).mockResolvedValue({});
    vi.mocked(roles.isAdminRole).mockReturnValue(true);
  });

  it("submits the bound fields only and never forwards api_base, timeout or max_retries", async () => {
    const user = userEvent.setup();
    renderPage();
    await openEditModal(user);

    await user.click(screen.getByRole("button", { name: "OK" }));

    await waitFor(() => expect(networking.updateSearchTool).toHaveBeenCalledTimes(1));
    const [token, toolId, payload] = vi.mocked(networking.updateSearchTool).mock.calls[0];
    expect(token).toBe("test-token");
    expect(toolId).toBe("tool-1");
    expect(payload).toStrictEqual({
      search_tool_name: "Perplexity Search",
      litellm_params: {
        search_provider: "perplexity",
        api_key: "sk-test-key",
        api_base: undefined,
        timeout: undefined,
        max_retries: undefined,
      },
      search_tool_info: { description: "Test description" },
    });
    expect(JSON.stringify(payload)).toBe(
      '{"search_tool_name":"Perplexity Search","litellm_params":{"search_provider":"perplexity","api_key":"sk-test-key"},"search_tool_info":{"description":"Test description"}}',
    );
  });

  it("carries an edited description through to search_tool_info", async () => {
    const user = userEvent.setup();
    renderPage();
    await openEditModal(user);

    await user.clear(screen.getByLabelText("Description"));
    fireEvent.change(screen.getByLabelText("Description"), { target: { value: "updated copy" } });
    await user.click(screen.getByRole("button", { name: "OK" }));

    await waitFor(() => expect(networking.updateSearchTool).toHaveBeenCalledTimes(1));
    expect(vi.mocked(networking.updateSearchTool).mock.calls[0][2]).toMatchObject({
      search_tool_info: { description: "updated copy" },
    });
  });

  it("drops search_tool_info entirely when the description is cleared", async () => {
    const user = userEvent.setup();
    renderPage();
    await openEditModal(user);

    await user.clear(screen.getByLabelText("Description"));
    await user.click(screen.getByRole("button", { name: "OK" }));

    await waitFor(() => expect(networking.updateSearchTool).toHaveBeenCalledTimes(1));
    const payload = vi.mocked(networking.updateSearchTool).mock.calls[0][2];
    expect(payload).toStrictEqual({
      search_tool_name: "Perplexity Search",
      litellm_params: {
        search_provider: "perplexity",
        api_key: "sk-test-key",
        api_base: undefined,
        timeout: undefined,
        max_retries: undefined,
      },
      search_tool_info: undefined,
    });
  });

  it("does not submit when Enter is pressed inside a modal field", async () => {
    const user = userEvent.setup();
    renderPage();
    await openEditModal(user);

    await user.type(screen.getByLabelText("Search Tool Name"), "{Enter}");

    await new Promise((resolve) => setTimeout(resolve, 200));
    expect(networking.updateSearchTool).not.toHaveBeenCalled();
  });

  it("blocks submit and shows the required message when the name is cleared", async () => {
    const user = userEvent.setup();
    renderPage();
    await openEditModal(user);

    await user.clear(screen.getByLabelText("Search Tool Name"));
    await user.click(screen.getByRole("button", { name: "OK" }));

    expect(await screen.findByText("Please enter a search tool name")).toBeInTheDocument();
    expect(networking.updateSearchTool).not.toHaveBeenCalled();
  });
  it("still edits a tool whose api_key and search_tool_info came back null", async () => {
    vi.mocked(networking.fetchSearchTools).mockResolvedValue({ search_tools: [toolWithNullServerFields] });
    const user = userEvent.setup();
    renderPage();
    await openEditModal(user);

    await user.click(screen.getByRole("button", { name: "OK" }));

    await waitFor(() => expect(networking.updateSearchTool).toHaveBeenCalledTimes(1));
    const payload = vi.mocked(networking.updateSearchTool).mock.calls[0][2];
    expect(payload).toStrictEqual({
      search_tool_name: "Perplexity Search",
      litellm_params: {
        search_provider: "perplexity",
        api_key: null,
        api_base: undefined,
        timeout: undefined,
        max_retries: undefined,
      },
      search_tool_info: undefined,
    });
  });
});
