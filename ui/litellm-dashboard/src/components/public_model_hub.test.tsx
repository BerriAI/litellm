import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, it, expect, vi, beforeAll, beforeEach, afterEach, type Mock } from "vitest";
import { render, screen, waitFor, within, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { flexRender, getCoreRowModel, useReactTable } from "@tanstack/react-table";
import type { OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { renderWithProviders } from "@/../tests/test-utils";
import PublicModelHub from "./public_model_hub";
import { AgentCard, getPublicMCPHubColumns, MCPServerData, ModelGroupInfo } from "./PublicModelHubTableColumns";

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

interface HubUrlOptions {
  searchParams?: string;
  onUrlUpdate?: OnUrlUpdateFunction;
}

const renderHub = (urlOptions: HubUrlOptions = {}) => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  return renderWithProviders(
    <QueryClientProvider client={client}>
      <PublicModelHub />
    </QueryClientProvider>,
    urlOptions,
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
      href: "http://localhost:3000/",
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

    fireEvent.change(screen.getByPlaceholderText("Search model names..."), { target: { value: "claude" } });

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

  it("keeps a deep-linked ?page= when the model page fails to load", async () => {
    const user = userEvent.setup();
    apiGetMock.mockRejectedValue(new Error("boom"));
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderHub({ searchParams: "?page=3", onUrlUpdate });
    expect(await screen.findByText(/Service unavailable/)).toBeInTheDocument();

    await user.click(screen.getByRole("tab", { name: "Skill Hub" }));

    await waitFor(() => expect(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.get("tab")).toBe("skills"));
    expect(onUrlUpdate.mock.calls.map(([update]) => update.searchParams.get("page"))).toEqual(
      onUrlUpdate.mock.calls.map(() => "3"),
    );
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

describe("PublicModelHub URL state", () => {
  const agentCard = (name: string, description: string, version: string, tag: string): AgentCard => ({
    protocolVersion: "0.3.0",
    name,
    description,
    url: `https://agents.example/${tag}`,
    version,
    defaultInputModes: ["text"],
    defaultOutputModes: ["text"],
    skills: [{ id: `${tag}-skill`, name: `${tag} skill`, description: "", tags: [tag] }],
  });
  const AGENTS = [
    agentCard("Billing Router", "routes billing questions", "1.0.0", "billing"),
    agentCard("Support Bot", "handles support tickets", "2.0.0", "support"),
  ];
  const mcpServer = (server_id: string, server_name: string, transport: string): MCPServerData => ({
    server_id,
    name: server_name,
    server_name,
    transport,
    auth_type: "none",
    mcp_info: { server_name, description: `${server_name} tools` },
  });
  const MCP_SERVERS = [mcpServer("server-1", "exa_test", "http"), mcpServer("server-2", "zeta-files", "sse")];
  const padded = (prefix: string, index: number) => `${prefix}-${String(index).padStart(2, "0")}`;
  const MANY_MCP_SERVERS = Array.from({ length: 60 }, (_, index) =>
    mcpServer(padded("id", index), padded("mcp", index), index < 30 ? "sse" : "http"),
  );
  const MANY_AGENTS = Array.from({ length: 60 }, (_, index) =>
    agentCard(padded("agent", index), "generated", "1.0.0", index < 30 ? "billing" : "support"),
  );

  const AGENT_ROWS = /^(Billing Router|Support Bot|agent-\d+)$/;
  const MCP_ROWS = /^(exa_test|zeta-files|mcp-\d+)$/;
  const MODEL_ROWS = /^(gpt-4|claude-3)$/;

  const hubMocks = async () => {
    const networkingModule = await import("./networking");
    return {
      agents: vi.mocked(networkingModule.agentHubPublicModelsCall),
      mcp: vi.mocked(networkingModule.mcpHubPublicServersCall),
    };
  };

  afterEach(async () => {
    const mocks = await hubMocks();
    mocks.agents.mockResolvedValue([]);
    mocks.mcp.mockResolvedValue([]);
  });

  const renderUrlHub = async (
    searchParams = "",
    agents: AgentCard[] = AGENTS,
    servers: MCPServerData[] = MCP_SERVERS,
  ) => {
    const mocks = await hubMocks();
    mocks.agents.mockResolvedValue(agents);
    mocks.mcp.mockResolvedValue(servers);
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderHub({ searchParams, onUrlUpdate });
    await screen.findByRole("tab", { name: "MCP Hub", hidden: true });
    return { user: userEvent.setup(), onUrlUpdate };
  };

  const lastUrl = (onUrlUpdate: Mock<OnUrlUpdateFunction>) => {
    const update = onUrlUpdate.mock.calls.at(-1)?.[0];
    if (!update) throw new Error("no URL update was emitted");
    return update;
  };

  const activePanel = () => screen.getByRole("tabpanel", { hidden: true });
  const rowNames = (names: RegExp) =>
    within(activePanel())
      .queryAllByRole("button", { name: names, hidden: true })
      .map((button) => button.textContent);
  const selectedTab = () => screen.getByRole("tab", { selected: true });
  const pageLabel = () => within(activePanel()).getByTestId("pagination-page");

  it("opens the hub tab named by ?tab=", async () => {
    await renderUrlHub("?tab=mcp");

    expect(selectedTab()).toHaveTextContent("MCP Hub");
    expect(rowNames(MCP_ROWS)).toEqual(["exa_test", "zeta-files"]);
  });

  it("writes the chosen hub tab to ?tab= and drops it for the Model Hub", async () => {
    const { user, onUrlUpdate } = await renderUrlHub();

    await user.click(screen.getByRole("tab", { name: "Agent Hub" }));
    await waitFor(() => expect(lastUrl(onUrlUpdate).searchParams.get("tab")).toBe("agents"));
    expect(rowNames(AGENT_ROWS)).toEqual(["Billing Router", "Support Bot"]);

    await user.click(screen.getByRole("tab", { name: "Model Hub" }));
    await waitFor(() => expect(lastUrl(onUrlUpdate).searchParams.has("tab")).toBe(false));
  });

  it("keeps ?tab=agents while the public agents are still loading", async () => {
    const mocks = await hubMocks();
    mocks.mcp.mockResolvedValue([]);
    let publishAgents: (agents: AgentCard[]) => void = () => {};
    mocks.agents.mockReturnValue(new Promise((resolve) => (publishAgents = resolve)));
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderHub({ searchParams: "?tab=agents", onUrlUpdate });
    await waitFor(() => expect(mocks.agents).toHaveBeenCalled());
    await waitFor(() => expect(modelCalls().length).toBeGreaterThan(0));
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(screen.queryByRole("tab", { name: "Agent Hub" })).not.toBeInTheDocument();

    publishAgents(AGENTS);

    expect(await screen.findByRole("tab", { name: "Agent Hub", selected: true })).toBeInTheDocument();
    expect(onUrlUpdate).not.toHaveBeenCalled();
  });

  it("falls back to the Model Hub and drops ?tab=agents when there are no public agents", async () => {
    const { onUrlUpdate } = await renderUrlHub("?tab=agents", []);

    await waitFor(() => expect(onUrlUpdate).toHaveBeenCalled());
    expect(lastUrl(onUrlUpdate).searchParams.has("tab")).toBe(false);
    expect(selectedTab()).toHaveTextContent("Model Hub");
    expect(screen.queryByRole("tab", { name: "Agent Hub" })).not.toBeInTheDocument();
  });

  it("opens the model named by ?model=", async () => {
    await renderUrlHub("?model=claude-3");

    const dialog = await screen.findByRole("dialog");
    expect(within(dialog).getByRole("heading", { name: "claude-3" })).toBeInTheDocument();
    expect(dialog).toHaveTextContent("Model Overview");
  });

  it("looks up a deep-linked model that is not on the current page", async () => {
    const deepLinked = model({ model_group: "o1-preview", providers: ["openai"] });
    respondWith(DEFAULT_MODELS);
    const listPage = apiGetMock.getMockImplementation();
    apiGetMock.mockImplementation((path: string, options?: { query?: QueryRecord }) =>
      options?.query?.q === "o1-preview"
        ? Promise.resolve({ data: [model({ model_group: "o1-preview-mini" }), deepLinked] })
        : listPage?.(path, options),
    );

    await renderUrlHub("?model=o1-preview");

    const dialog = await screen.findByRole("dialog");
    expect(within(dialog).getByRole("heading", { name: "o1-preview" })).toBeInTheDocument();
    expect(apiGetMock).toHaveBeenCalledWith(
      MODEL_HUB_PATH,
      expect.objectContaining({ query: { q: "o1-preview", page_size: 100 } }),
    );
    expect(rowNames(MODEL_ROWS)).toEqual(["gpt-4", "claude-3"]);
  });

  it("does not look up a deep-linked model that is already on the page", async () => {
    await renderUrlHub("?model=gpt-4");

    await screen.findByRole("dialog");
    expect(modelQueries().every((query) => query.q === undefined)).toBe(true);
  });

  it("pushes the clicked model into ?model= and clears it when the dialog closes", async () => {
    const { user, onUrlUpdate } = await renderUrlHub();
    await screen.findByText("gpt-4");

    await user.click(within(activePanel()).getByRole("button", { name: "gpt-4" }));

    await waitFor(() => expect(lastUrl(onUrlUpdate).searchParams.get("model")).toBe("gpt-4"));
    expect(lastUrl(onUrlUpdate).options.history).toBe("push");
    const dialog = await screen.findByRole("dialog");
    expect(within(dialog).getByRole("heading", { name: "gpt-4" })).toBeInTheDocument();

    await user.click(within(dialog).getByRole("button", { name: /close/i }));

    await waitFor(() => expect(lastUrl(onUrlUpdate).searchParams.has("model")).toBe(false));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
  });

  it("opens the agent named by ?agent=", async () => {
    await renderUrlHub("?tab=agents&agent=Support%20Bot");

    const dialog = await screen.findByRole("dialog");
    expect(within(dialog).getByRole("heading", { name: "Support Bot" })).toBeInTheDocument();
    expect(dialog).toHaveTextContent("Agent Overview");
  });

  it("pushes the clicked agent name into ?agent= and clears it when the dialog closes", async () => {
    const { user, onUrlUpdate } = await renderUrlHub("?tab=agents");

    await user.click(within(activePanel()).getByRole("button", { name: "Billing Router" }));

    await waitFor(() => expect(lastUrl(onUrlUpdate).searchParams.get("agent")).toBe("Billing Router"));
    expect(lastUrl(onUrlUpdate).options.history).toBe("push");
    const dialog = await screen.findByRole("dialog");
    expect(within(dialog).getByRole("heading", { name: "Billing Router" })).toBeInTheDocument();

    await user.click(within(dialog).getByRole("button", { name: /close/i }));

    await waitFor(() => expect(lastUrl(onUrlUpdate).searchParams.has("agent")).toBe(false));
    expect(lastUrl(onUrlUpdate).options.history).toBe("push");
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
  });

  it("filters agents by agent_q and writes the search back", async () => {
    const { onUrlUpdate } = await renderUrlHub("?tab=agents&agent_q=support");
    const search = screen.getByPlaceholderText("Search agent names or descriptions...");
    expect(search).toHaveValue("support");
    expect(rowNames(AGENT_ROWS)).toEqual(["Support Bot"]);

    fireEvent.change(search, { target: { value: "billing" } });

    await waitFor(() => expect(lastUrl(onUrlUpdate).searchParams.get("agent_q")).toBe("billing"));
    expect(rowNames(AGENT_ROWS)).toEqual(["Billing Router"]);
  });

  it("filters agents by the skills in agent_skills", async () => {
    await renderUrlHub("?tab=agents&agent_skills=support");

    expect(rowNames(AGENT_ROWS)).toEqual(["Support Bot"]);
    expect(within(activePanel()).getByLabelText("support")).toBeInTheDocument();
  });

  it("writes the agent table page to agent_page", async () => {
    const { user, onUrlUpdate } = await renderUrlHub("?tab=agents", MANY_AGENTS);
    expect(pageLabel()).toHaveTextContent("Page 1 of 3");

    await user.click(within(activePanel()).getByTestId("pagination-next"));

    await waitFor(() => expect(lastUrl(onUrlUpdate).searchParams.get("agent_page")).toBe("2"));
    expect(pageLabel()).toHaveTextContent("Page 2 of 3");
    expect(rowNames(AGENT_ROWS)[0]).toBe("agent-25");
  });

  it("writes picked skills to agent_skills and returns the agent table to its first page", async () => {
    const { user, onUrlUpdate } = await renderUrlHub("?tab=agents&agent_page=2", MANY_AGENTS);
    expect(pageLabel()).toHaveTextContent("Page 2 of 3");

    await user.click(screen.getByPlaceholderText("Select skills"));
    await user.click(await screen.findByRole("option", { name: "billing" }));

    await waitFor(() => expect(lastUrl(onUrlUpdate).searchParams.getAll("agent_skills")).toEqual(["billing"]));
    expect(lastUrl(onUrlUpdate).searchParams.has("agent_page")).toBe(false);
    expect(pageLabel()).toHaveTextContent("Page 1 of 2");
    expect(rowNames(AGENT_ROWS)[0]).toBe("agent-00");
  });

  it("sorts agents from agent_ keys and writes header clicks back", async () => {
    const { user, onUrlUpdate } = await renderUrlHub("?tab=agents&agent_sort_order=desc");
    expect(rowNames(AGENT_ROWS)).toEqual(["Support Bot", "Billing Router"]);

    await user.click(within(activePanel()).getByTestId("sort-header-version"));

    await waitFor(() => expect(lastUrl(onUrlUpdate).searchParams.get("agent_sort_by")).toBe("version"));
    expect(rowNames(AGENT_ROWS)).toEqual(["Billing Router", "Support Bot"]);
  });

  it("opens the MCP server whose id is in ?mcp=", async () => {
    await renderUrlHub("?tab=mcp&mcp=server-2");

    const dialog = await screen.findByRole("dialog");
    expect(within(dialog).getByRole("heading", { name: "zeta-files" })).toBeInTheDocument();
    expect(dialog).toHaveTextContent("Server Overview");
  });

  it("pushes the clicked MCP server id into ?mcp=", async () => {
    const { user, onUrlUpdate } = await renderUrlHub("?tab=mcp");

    await user.click(within(activePanel()).getByRole("button", { name: "exa_test" }));

    await waitFor(() => expect(lastUrl(onUrlUpdate).searchParams.get("mcp")).toBe("server-1"));
    expect(lastUrl(onUrlUpdate).options.history).toBe("push");
    expect(await screen.findByText("Server Overview")).toBeInTheDocument();
  });

  it("filters MCP servers by mcp_q from the URL", async () => {
    await renderUrlHub("?tab=mcp&mcp_q=zeta");
    expect(screen.getByPlaceholderText("Search MCP server names or descriptions...")).toHaveValue("zeta");
    expect(rowNames(MCP_ROWS)).toEqual(["zeta-files"]);
  });

  it("filters MCP servers by the transports in mcp_transport", async () => {
    await renderUrlHub("?tab=mcp&mcp_transport=http");

    expect(rowNames(MCP_ROWS)).toEqual(["exa_test"]);
  });

  it("writes the MCP table page and page size to mcp_page and mcp_page_size", async () => {
    const { user, onUrlUpdate } = await renderUrlHub("?tab=mcp", AGENTS, MANY_MCP_SERVERS);
    expect(pageLabel()).toHaveTextContent("Page 1 of 3");

    await user.click(within(activePanel()).getByTestId("pagination-next"));

    await waitFor(() => expect(lastUrl(onUrlUpdate).searchParams.get("mcp_page")).toBe("2"));
    expect(pageLabel()).toHaveTextContent("Page 2 of 3");
    expect(rowNames(MCP_ROWS)[0]).toBe("mcp-25");

    await user.click(within(activePanel()).getByTestId("pagination-page-size"));
    await user.click(await screen.findByRole("option", { name: "50" }));

    await waitFor(() => expect(lastUrl(onUrlUpdate).searchParams.get("mcp_page_size")).toBe("50"));
    expect(pageLabel()).toHaveTextContent("of 2");
  });

  it("reads the MCP page from mcp_page and returns to the first page when a transport is picked", async () => {
    const { user, onUrlUpdate } = await renderUrlHub("?tab=mcp&mcp_page=2", AGENTS, MANY_MCP_SERVERS);
    expect(pageLabel()).toHaveTextContent("Page 2 of 3");
    expect(rowNames(MCP_ROWS)[0]).toBe("mcp-25");

    await user.click(screen.getByPlaceholderText("Select transport types"));
    await user.click(await screen.findByRole("option", { name: "sse" }));

    await waitFor(() => expect(lastUrl(onUrlUpdate).searchParams.getAll("mcp_transport")).toEqual(["sse"]));
    expect(lastUrl(onUrlUpdate).searchParams.has("mcp_page")).toBe(false);
    expect(pageLabel()).toHaveTextContent("Page 1 of 2");
    expect(rowNames(MCP_ROWS)[0]).toBe("mcp-00");
  });

  it("writes the MCP search, transport and sort to mcp_ keys", async () => {
    const { user, onUrlUpdate } = await renderUrlHub("?tab=mcp&mcp_sort_order=desc");
    expect(rowNames(MCP_ROWS)).toEqual(["zeta-files", "exa_test"]);

    fireEvent.change(screen.getByPlaceholderText("Search MCP server names or descriptions..."), {
      target: { value: "e" },
    });
    await waitFor(() => expect(lastUrl(onUrlUpdate).searchParams.get("mcp_q")).toBe("e"));

    await user.click(screen.getByPlaceholderText("Select transport types"));
    await user.click(await screen.findByRole("option", { name: "sse" }));
    await waitFor(() => expect(lastUrl(onUrlUpdate).searchParams.getAll("mcp_transport")).toEqual(["sse"]));
    expect(rowNames(MCP_ROWS)).toEqual(["zeta-files"]);

    await user.keyboard("{Escape}");
    await user.click(within(activePanel()).getByTestId("sort-header-transport"));
    await waitFor(() => expect(lastUrl(onUrlUpdate).searchParams.get("mcp_sort_by")).toBe("transport"));
    expect(lastUrl(onUrlUpdate).searchParams.get("mcp_q")).toBe("e");
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
