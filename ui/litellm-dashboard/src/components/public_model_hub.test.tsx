import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, it, expect, vi, beforeAll, beforeEach } from "vitest";
import { render, screen, waitFor, within, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { flexRender, getCoreRowModel, useReactTable } from "@tanstack/react-table";
import PublicModelHub from "./public_model_hub";
import { getPublicMCPHubColumns, MCPServerData, ModelGroupInfo } from "./PublicModelHubTableColumns";

const { apiGetMock } = vi.hoisted(() => ({ apiGetMock: vi.fn() }));

vi.mock("next/navigation", () => ({
  useRouter: vi.fn(() => ({
    replace: vi.fn(),
    push: vi.fn(),
    refresh: vi.fn(),
  })),
}));

vi.mock("./networking", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./networking")>();
  return {
    ...actual,
    apiClient: { ...actual.apiClient, get: apiGetMock },
    modelHubPublicModelsCall: vi.fn().mockResolvedValue([]),
    getPublicModelHubInfo: vi.fn().mockResolvedValue({
      docs_title: "LiteLLM Gateway",
      custom_docs_description: null,
      litellm_version: "1.0.0",
      useful_links: {},
    }),
    agentHubPublicModelsCall: vi.fn().mockResolvedValue([]),
    mcpHubPublicServersCall: vi.fn().mockResolvedValue([]),
    skillHubPublicCall: vi.fn().mockResolvedValue({ plugins: [] }),
    getUiConfig: vi.fn().mockResolvedValue({}),
  };
});

vi.mock("./navbar", () => ({
  default: vi.fn(() => <div data-testid="navbar">Navbar Component</div>),
}));

const MODEL_HUB_PATH = "/public/v1/model_hub";

const FACET_VALUES: Record<string, string[]> = {
  [`${MODEL_HUB_PATH}/providers`]: ["anthropic", "openai"],
  [`${MODEL_HUB_PATH}/modes`]: ["chat", "embedding"],
  [`${MODEL_HUB_PATH}/features`]: ["function_calling", "vision"],
};

const MODEL_DEFAULTS = {
  providers: ["openai"],
  mode: "chat",
  supports_function_calling: false,
  supports_vision: false,
  supports_parallel_function_calling: false,
};

const model = (overrides: Partial<ModelGroupInfo> & { model_group: string }): ModelGroupInfo => ({
  ...MODEL_DEFAULTS,
  ...overrides,
});

const DEFAULT_MODELS = [model({ model_group: "gpt-4" }), model({ model_group: "claude-3", providers: ["anthropic"] })];

const respondWith = (rows: ModelGroupInfo[], totalCount: number = rows.length, pageSize: number = 50) =>
  apiGetMock.mockImplementation((path: string) => {
    const facet = FACET_VALUES[path];
    if (facet) {
      return Promise.resolve({
        data: facet,
        meta: { page: 1, page_size: 100, has_more: false },
        links: { self: path, prev: null, next: null },
      });
    }
    return Promise.resolve({
      data: rows,
      meta: {
        total_count: totalCount,
        page: 1,
        page_size: pageSize,
        total_pages: Math.max(Math.ceil(totalCount / pageSize), 1),
      },
      links: { self: MODEL_HUB_PATH, first: MODEL_HUB_PATH, prev: null, next: null, last: MODEL_HUB_PATH },
    });
  });

type QueryRecord = Record<string, string | number>;

const modelCalls = () => apiGetMock.mock.calls.filter((call) => call[0] === MODEL_HUB_PATH);
const facetPaths = (): string[] =>
  apiGetMock.mock.calls.map((call) => String(call[0])).filter((path) => path.startsWith(`${MODEL_HUB_PATH}/`));
const modelQueries = (): QueryRecord[] => modelCalls().map((call) => (call[1] as { query: QueryRecord }).query);
const lastModelQuery = (): QueryRecord => modelQueries()[modelQueries().length - 1];

const renderHub = () => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  return render(
    <QueryClientProvider client={client}>
      <PublicModelHub />
    </QueryClientProvider>,
  );
};

beforeAll(() => {
  Object.defineProperty(window, "matchMedia", {
    writable: true,
    value: (query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: () => {},
      removeListener: () => {},
      addEventListener: () => {},
      removeEventListener: () => {},
      dispatchEvent: () => false,
    }),
  });
});

beforeEach(() => {
  vi.clearAllMocks();
  respondWith(DEFAULT_MODELS);
  Storage.prototype.getItem = vi.fn(() => "false");
  Storage.prototype.setItem = vi.fn();
  Object.defineProperty(window, "location", {
    writable: true,
    value: {
      pathname: "/",
      origin: "http://localhost:3000",
    },
  });
});

describe("PublicModelHub", () => {
  it("renders", () => {
    const { container } = renderHub();
    expect(container).toBeInTheDocument();
  });

  it("loads the first page of models from the paginated public endpoint", async () => {
    renderHub();

    expect(await screen.findByText("gpt-4")).toBeInTheDocument();
    expect(modelCalls()[0][0]).toBe(MODEL_HUB_PATH);
    expect(modelQueries()[0]).toEqual({ page: 1, page_size: 50, sort: "model_group" });
  });

  it("waits for the resolved proxy base url before asking for a page", async () => {
    const networkingModule = await import("./networking");
    let publishConfig: () => void = () => {};
    vi.mocked(networkingModule.getUiConfig).mockReturnValueOnce(
      new Promise((resolve) => {
        publishConfig = () => resolve({} as Awaited<ReturnType<typeof networkingModule.getUiConfig>>);
      }),
    );

    renderHub();
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(modelCalls()).toHaveLength(0);

    publishConfig();

    await waitFor(() => expect(modelCalls().length).toBeGreaterThan(0));
  });

  it("stops calling the unpaginated public model hub route", async () => {
    const networkingModule = await import("./networking");
    renderHub();

    await waitFor(() => expect(apiGetMock).toHaveBeenCalled());
    expect(networkingModule.modelHubPublicModelsCall).not.toHaveBeenCalled();
  });

  it("counts the whole catalogue from the response meta, not the rows on screen", async () => {
    respondWith(DEFAULT_MODELS, 300);
    renderHub();

    await screen.findByText("gpt-4");
    expect(screen.getByTestId("pagination-range")).toHaveTextContent("of 300");
    expect(screen.getByTestId("pagination-page")).toHaveTextContent("Page 1 of 6");
  });

  it("asks the server for the next page", async () => {
    const user = userEvent.setup();
    respondWith(DEFAULT_MODELS, 300);
    renderHub();
    await screen.findByText("gpt-4");

    await user.click(screen.getByTestId("pagination-next"));

    await waitFor(() => expect(lastModelQuery().page).toBe(2));
    expect(lastModelQuery().page_size).toBe(50);
  });

  it("asks the server for a different page size", async () => {
    const user = userEvent.setup();
    respondWith(DEFAULT_MODELS, 300);
    renderHub();
    await screen.findByText("gpt-4");

    await user.click(screen.getByTestId("pagination-page-size"));
    await user.click(await screen.findByRole("option", { name: "25" }));

    await waitFor(() => expect(lastModelQuery().page_size).toBe(25));
  });

  it("asks the server to sort, in the sort form the endpoint accepts", async () => {
    const user = userEvent.setup();
    renderHub();
    await screen.findByText("gpt-4");

    await user.click(screen.getByTestId("sort-header-model_group"));
    await waitFor(() => expect(lastModelQuery().sort).toBe("-model_group"));

    await user.click(screen.getByTestId("sort-header-input_cost_per_token"));
    await waitFor(() => expect(lastModelQuery().sort).toBe("-input_cost_per_token"));
  });

  it("renders the page in the order the server sent it, without re-sorting locally", async () => {
    const user = userEvent.setup();
    respondWith([model({ model_group: "alpha-model" }), model({ model_group: "zeta-model" })], 300);
    renderHub();
    await screen.findByText("alpha-model");

    await user.click(screen.getByTestId("sort-header-model_group"));
    await waitFor(() => expect(lastModelQuery().sort).toBe("-model_group"));

    const rendered = screen.getAllByText(/-model$/).map((cell) => cell.textContent);
    expect(rendered).toEqual(["alpha-model", "zeta-model"]);
  });

  it("offers sorting on exactly the fields the endpoint accepts", async () => {
    renderHub();
    await screen.findByText("gpt-4");

    const sortable = screen
      .getAllByTestId(/^sort-header-/)
      .map((header) => header.getAttribute("data-testid")?.replace("sort-header-", ""));

    expect(sortable.sort()).toEqual([
      "input_cost_per_token",
      "max_input_tokens",
      "max_output_tokens",
      "mode",
      "model_group",
      "output_cost_per_token",
      "providers",
      "rpm",
    ]);
    expect(screen.getByText("Health Status")).toBeInTheDocument();
    expect(screen.queryByTestId("sort-header-health_status")).not.toBeInTheDocument();
  });

  it("searches on the server and returns to the first page", async () => {
    const user = userEvent.setup();
    respondWith(DEFAULT_MODELS, 300);
    renderHub();
    await screen.findByText("gpt-4");

    await user.click(screen.getByTestId("pagination-next"));
    await waitFor(() => expect(lastModelQuery().page).toBe(2));

    await user.type(screen.getByPlaceholderText("Search model names..."), "claude");

    await waitFor(() => expect(lastModelQuery().q).toBe("claude"));
    expect(lastModelQuery().page).toBe(1);
  });

  it("filters by mode with the endpoint's in operator", async () => {
    const user = userEvent.setup();
    renderHub();
    await screen.findByText("gpt-4");

    await user.click(screen.getByPlaceholderText("Select modes"));
    await user.click(await screen.findByRole("option", { name: "embedding" }));

    await waitFor(() => expect(lastModelQuery()["filter[mode][in]"]).toBe("embedding"));
  });

  it("filters by several providers at once, and returns to the first page", async () => {
    const user = userEvent.setup();
    respondWith(DEFAULT_MODELS, 300);
    renderHub();
    await screen.findByText("gpt-4");

    await user.click(screen.getByTestId("pagination-next"));
    await waitFor(() => expect(lastModelQuery().page).toBe(2));

    await user.click(screen.getByPlaceholderText("Select providers"));
    await user.click(await screen.findByRole("option", { name: /anthropic/i }));
    await waitFor(() => expect(lastModelQuery()["filter[providers][in]"]).toBe("anthropic"));
    expect(lastModelQuery().page).toBe(1);

    await user.click(await screen.findByRole("option", { name: /openai/i }));

    await waitFor(() => expect(lastModelQuery()["filter[providers][in]"]).toBe("anthropic,openai"));
  });

  it("filters by feature, which the table could not do while it paged", async () => {
    const user = userEvent.setup();
    renderHub();
    await screen.findByText("gpt-4");

    await user.click(screen.getByPlaceholderText("Select features"));
    await user.click(await screen.findByRole("option", { name: "Vision" }));

    await waitFor(() => expect(lastModelQuery()["filter[features][in]"]).toBe("vision"));
  });

  it("offers the filter values the route reports, not the ones on the page", async () => {
    respondWith([model({ model_group: "gpt-4" })], 1);
    renderHub();
    await screen.findByText("gpt-4");

    await waitFor(() => expect(facetPaths()).toContain(`${MODEL_HUB_PATH}/providers`));
    expect(facetPaths()).toEqual(expect.arrayContaining([`${MODEL_HUB_PATH}/modes`, `${MODEL_HUB_PATH}/features`]));
  });

  it("displays health status correctly for models with health check information", async () => {
    respondWith([
      {
        ...MODEL_DEFAULTS,
        model_group: "gpt-4",
        health_status: "healthy",
        health_response_time: 150.5,
        health_checked_at: "2024-01-15T10:30:00Z",
      },
      {
        ...MODEL_DEFAULTS,
        model_group: "claude-3",
        providers: ["anthropic"],
        health_status: "unhealthy",
        health_response_time: 5000.0,
        health_checked_at: "2024-01-15T10:25:00Z",
      },
      model({ model_group: "gpt-3.5-turbo" }),
    ]);

    renderHub();

    await waitFor(() => {
      expect(screen.getByText("gpt-4")).toBeInTheDocument();
    });

    await waitFor(() => {
      const gpt4Row = screen.getByText("gpt-4").closest("tr");
      expect(gpt4Row).toBeInTheDocument();
      expect(within(gpt4Row as HTMLElement).getByText("healthy")).toBeInTheDocument();
    });

    await waitFor(() => {
      const claude3Row = screen.getByText("claude-3").closest("tr");
      expect(claude3Row).toBeInTheDocument();
      expect(within(claude3Row as HTMLElement).getByText("unhealthy")).toBeInTheDocument();
    });

    await waitFor(() => {
      const gpt35Row = screen.getByText("gpt-3.5-turbo").closest("tr");
      expect(gpt35Row).toBeInTheDocument();
      expect(within(gpt35Row as HTMLElement).getByText("Unknown")).toBeInTheDocument();
    });
  });

  it("shows no models when the search has no matches (LIT-5230 regression)", async () => {
    renderHub();
    expect(await screen.findByText("gpt-4")).toBeInTheDocument();

    respondWith([], 0);
    fireEvent.change(screen.getByPlaceholderText("Search model names..."), { target: { value: "zzzz" } });

    await waitFor(() => {
      expect(screen.queryByText("gpt-4")).not.toBeInTheDocument();
      expect(screen.queryByText("claude-3")).not.toBeInTheDocument();
      expect(screen.getByText("No matching models")).toBeInTheDocument();
    });
  });

  it("reports the proxy as unavailable when the model page fails to load", async () => {
    apiGetMock.mockRejectedValue(new Error("boom"));

    renderHub();

    expect(await screen.findByText(/Service unavailable/)).toBeInTheDocument();
  });

  it("keeps the page usable when the response carries no rows", async () => {
    respondWith([], 0);

    renderHub();

    await waitFor(() => {
      expect(screen.getByTestId("navbar")).toBeInTheDocument();
      expect(screen.getByText("Model Hub")).toBeInTheDocument();
      expect(screen.getByText("No models available")).toBeInTheDocument();
    });
  });
});

const PUBLIC_SERVER_URL = "https://mcp.exa.ai/mcp";

const mockMcpServer: MCPServerData = {
  server_id: "server-1",
  name: "exa_test",
  server_name: "exa_test",
  url: PUBLIC_SERVER_URL,
  transport: "http",
  auth_type: "none",
  mcp_info: { server_name: "exa_test", description: "Fast, intelligent web search and web crawling" },
};

function PublicMcpTestTable({ data }: { data: MCPServerData[] }) {
  const columns = getPublicMCPHubColumns({ onServerClick: vi.fn() });
  const table = useReactTable({ data, columns, getCoreRowModel: getCoreRowModel() });

  return (
    <table>
      <thead>
        {table.getHeaderGroups().map((hg) => (
          <tr key={hg.id}>
            {hg.headers.map((h) => (
              <th key={h.id}>{flexRender(h.column.columnDef.header, h.getContext())}</th>
            ))}
          </tr>
        ))}
      </thead>
      <tbody>
        {table.getRowModel().rows.map((row) => (
          <tr key={row.id}>
            {row.getVisibleCells().map((cell) => (
              <td key={cell.id}>{flexRender(cell.column.columnDef.cell, cell.getContext())}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

describe("publicMCPHubColumns", () => {
  it("keeps the non-sensitive columns", () => {
    render(<PublicMcpTestTable data={[mockMcpServer]} />);
    expect(screen.getByText("Server Name")).toBeInTheDocument();
    expect(screen.getByText("Transport")).toBeInTheDocument();
    expect(screen.getByText("Auth Type")).toBeInTheDocument();
  });

  it("does not expose a URL column header", () => {
    render(<PublicMcpTestTable data={[mockMcpServer]} />);
    expect(screen.queryByText("URL")).not.toBeInTheDocument();
    const columns = getPublicMCPHubColumns({ onServerClick: vi.fn() });
    expect(columns.some((c) => c.header === "URL" || c.meta?.title === "URL")).toBe(false);
  });

  it("does not render the server url anywhere in the table", () => {
    render(<PublicMcpTestTable data={[mockMcpServer]} />);
    expect(screen.queryByText(PUBLIC_SERVER_URL)).not.toBeInTheDocument();
  });
});

describe("public hub MCP details modal", () => {
  it("does not show the upstream url when a server is opened", async () => {
    const networkingModule = await import("./networking");
    vi.mocked(networkingModule.mcpHubPublicServersCall).mockResolvedValue([mockMcpServer]);

    renderHub();

    fireEvent.click(await screen.findByRole("tab", { name: /MCP Hub/i }));
    fireEvent.click(await screen.findByRole("button", { name: "exa_test" }));

    // "Server Overview" only exists inside the opened MCP details modal,
    // so finding it proves the modal rendered and the url assertion is not vacuous.
    await screen.findByText("Server Overview");
    expect(screen.queryByText(PUBLIC_SERVER_URL)).not.toBeInTheDocument();
  });

  it("closes the server details modal from its close control", async () => {
    const networkingModule = await import("./networking");
    vi.mocked(networkingModule.mcpHubPublicServersCall).mockResolvedValue([mockMcpServer]);

    renderHub();

    fireEvent.click(await screen.findByRole("tab", { name: /MCP Hub/i }));
    fireEvent.click(await screen.findByRole("button", { name: "exa_test" }));
    await screen.findByText("Server Overview");

    fireEvent.click(screen.getByRole("button", { name: /close/i }));

    await waitFor(() => expect(screen.queryByText("Server Overview")).not.toBeInTheDocument());
  });
});
