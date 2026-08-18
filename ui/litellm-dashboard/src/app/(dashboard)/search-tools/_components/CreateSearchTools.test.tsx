import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import * as networking from "@/components/networking";
import CreateSearchTool, { SearchProviderLabel } from "./CreateSearchTools";

vi.mock("@/components/networking", () => ({
  createSearchTool: vi.fn(),
  fetchAvailableSearchProviders: vi.fn(),
}));

vi.mock("./SearchConnectionTest", () => {
  const SearchConnectionTest = () => <div>Running connection test</div>;
  return { default: SearchConnectionTest };
});

describe("SearchProviderLabel", () => {
  it("renders the tavily logo from the static bundle, untouched by server-root prefixing", () => {
    render(<SearchProviderLabel providerName="tavily" displayName="Tavily" />);
    const img = screen.getByRole("img", { name: "Tavily logo" });
    expect(img).toHaveAttribute("src", "/_next/static/media/tavily.png");
  });

  it("renders the exa_ai logo file for the exa_ai slug", () => {
    render(<SearchProviderLabel providerName="exa_ai" displayName="Exa AI" />);
    const img = screen.getByRole("img", { name: "Exa AI logo" });
    expect(img).toHaveAttribute("src", expect.stringContaining("exa_ai.png"));
  });

  it("renders the google_pse logo file for the google_pse slug", () => {
    render(<SearchProviderLabel providerName="google_pse" displayName="Google PSE" />);
    expect(screen.getByRole("img", { name: "Google PSE logo" })).toHaveAttribute(
      "src",
      expect.stringContaining("google_pse.png"),
    );
  });

  it("falls back to a letter avatar for a provider with no bundled logo", () => {
    render(<SearchProviderLabel providerName="brave" displayName="Brave Search" />);
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
    expect(screen.getByText("B")).toBeInTheDocument();
    expect(screen.getByText("Brave Search")).toBeInTheDocument();
  });

  it("does not guess a legacy /ui/assets/logos/<slug>.png url for unknown providers", () => {
    const { container } = render(<SearchProviderLabel providerName="searxng" displayName="SearXNG" />);
    expect(container.querySelector("img")).toBeNull();
  });
});

describe("CreateSearchTool", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(networking.fetchAvailableSearchProviders).mockResolvedValue({
      providers: [{ provider_name: "tavily", ui_friendly_name: "Tavily" }],
    });
  });

  const renderModal = () => {
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    return render(
      <QueryClientProvider client={queryClient}>
        <CreateSearchTool
          userRole="Admin"
          accessToken="sk-test"
          onCreateSuccess={vi.fn()}
          isModalVisible={true}
          setModalVisible={vi.fn()}
        />
      </QueryClientProvider>,
    );
  };

  it("opens the connection test without creating the tool when Test Connection is clicked on a valid form", async () => {
    const user = userEvent.setup();
    renderModal();

    await user.type(screen.getByLabelText(/Search Tool Name/), "tavily-search");
    fireEvent.mouseDown(screen.getByRole("combobox"));
    await user.click(await screen.findByText("Tavily"));
    await user.type(screen.getByLabelText(/API Key/), "tvly-secret");

    await user.click(screen.getByRole("button", { name: "Test Connection" }));

    expect(await screen.findByText("Running connection test")).toBeInTheDocument();
    expect(networking.createSearchTool).not.toHaveBeenCalled();
  });
});
