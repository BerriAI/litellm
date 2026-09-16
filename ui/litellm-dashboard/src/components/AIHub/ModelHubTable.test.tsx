import * as networking from "@/components/networking";
import userEvent from "@testing-library/user-event";
import type { OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { afterEach, beforeEach, describe, expect, it, Mock, vi } from "vitest";
import { act, fireEvent, renderWithProviders, screen, waitFor, within } from "../../../tests/test-utils";
import ModelHubTable from "./ModelHubTable";

const mockUseUISettings = vi.hoisted(() => vi.fn());
const mockGetCookie = vi.hoisted(() => vi.fn());
const mockCheckTokenValidity = vi.hoisted(() => vi.fn());
const mockRouterReplace = vi.hoisted(() => vi.fn());
const mockLocationReplace = vi.hoisted(() => vi.fn());

vi.mock("@/components/networking", () => ({
  getUiConfig: vi.fn(),
  modelHubPublicModelsCall: vi.fn(),
  modelHubCall: vi.fn(),
  getConfigFieldSetting: vi.fn(),
  getProxyBaseUrl: vi.fn(() => "http://localhost:4000"),
  getAgentsList: vi.fn(),
  fetchMCPServers: vi.fn(),
  getUiSettings: vi.fn(),
  getClaudeCodeMarketplace: vi.fn(),
  getClaudeCodePluginsList: vi.fn(() => Promise.resolve({ plugins: [] })),
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    replace: mockRouterReplace,
  }),
}));

vi.mock("@/components/public_model_hub", () => ({
  default: () => <div>Public Model Hub</div>,
}));

vi.mock("@/components/AIHub/forms/MakeAgentPublicForm", () => ({
  default: ({ visible, onSuccess }: { visible: boolean; onSuccess: () => void }) =>
    visible ? <button onClick={onSuccess}>Confirm agent publish</button> : null,
}));

vi.mock("@/app/(dashboard)/hooks/uiSettings/useUISettings", () => ({
  useUISettings: mockUseUISettings,
}));

vi.mock("@/utils/cookieUtils", () => ({
  getCookie: mockGetCookie,
}));

vi.mock("@/utils/jwtUtils", () => ({
  checkTokenValidity: mockCheckTokenValidity,
}));

describe("ModelHubTable", () => {
  const originalLocation = window.location;

  beforeEach(() => {
    Object.defineProperty(window, "location", {
      value: {
        href: "http://localhost:4000/ui/model_hub_table",
        origin: "http://localhost:4000",
        hostname: "localhost",
        pathname: "/ui/model_hub_table",
        search: "",
        protocol: "http:",
        replace: mockLocationReplace,
      },
      writable: true,
    });
  });

  afterEach(() => {
    Object.defineProperty(window, "location", {
      value: originalLocation,
      writable: true,
    });
    vi.clearAllMocks();
  });

  // Reusable helper function to setup mocks for auth redirect tests
  const setupAuthRedirectTest = (requireAuth: boolean, tokenValue: string | null, isTokenValid: boolean) => {
    mockUseUISettings.mockReturnValue({
      data: {
        values: {
          require_auth_for_public_ai_hub: requireAuth,
        },
      },
      isLoading: false,
    });
    mockGetCookie.mockReturnValue(tokenValue);
    mockCheckTokenValidity.mockReturnValue(isTokenValid);
    mockRouterReplace.mockClear();
    mockLocationReplace.mockClear();

    // Setup other required mocks
    vi.mocked(networking.getUiConfig).mockResolvedValue({
      server_root_path: "/",
      proxy_base_url: "http://localhost:4000",
      auto_redirect_to_sso: false,
      admin_ui_disabled: false,
      sso_configured: false,
    });
    vi.mocked(networking.modelHubPublicModelsCall).mockResolvedValue([]);
    vi.mocked(networking.getUiSettings).mockResolvedValue({
      values: {
        require_auth_for_public_ai_hub: requireAuth,
      },
    });
  };

  // Reusable test function for auth redirect scenarios
  const testAuthRedirect = (
    requireAuth: boolean,
    tokenValue: string | null,
    isTokenValid: boolean,
    shouldRedirect: boolean,
    description: string,
  ) => {
    it(description, async () => {
      setupAuthRedirectTest(requireAuth, tokenValue, isTokenValid);

      renderWithProviders(<ModelHubTable accessToken={null} publicPage={true} premiumUser={false} userRole={null} />);

      await waitFor(() => {
        if (shouldRedirect) {
          expect(mockLocationReplace).toHaveBeenCalledWith("http://localhost:4000/ui/login/");
          expect(mockRouterReplace).not.toHaveBeenCalled();
        } else {
          expect(mockLocationReplace).not.toHaveBeenCalled();
        }
      });
    });
  };

  it("should render", async () => {
    vi.mocked(networking.modelHubCall).mockResolvedValue({
      data: [],
    });
    vi.mocked(networking.getConfigFieldSetting).mockResolvedValue({
      field_value: false,
    });
    vi.mocked(networking.getAgentsList).mockResolvedValue({
      agents: [],
    });
    vi.mocked(networking.fetchMCPServers).mockResolvedValue([]);
    vi.mocked(networking.getUiSettings).mockResolvedValue({
      values: {},
    });
    mockUseUISettings.mockReturnValue({
      data: { values: {} },
      isLoading: false,
    });

    renderWithProviders(
      <ModelHubTable accessToken="test-token" publicPage={false} premiumUser={false} userRole={null} />,
    );

    await waitFor(() => {
      expect(screen.getByText("AI Hub")).toBeInTheDocument();
    });
  });

  it("should resolve loading to the empty state when there is no access token on the admin page", async () => {
    vi.mocked(networking.getUiSettings).mockResolvedValue({
      values: {},
    });
    mockUseUISettings.mockReturnValue({
      data: { values: {} },
      isLoading: false,
    });

    renderWithProviders(<ModelHubTable accessToken={null} publicPage={false} premiumUser={false} userRole={null} />);

    expect(await screen.findByText("No models yet")).toBeInTheDocument();
    expect(networking.modelHubCall).not.toHaveBeenCalled();
  });

  it("should call getUiConfig before modelHubPublicModelsCall when publicPage is true", async () => {
    const getUiConfigMock = vi.mocked(networking.getUiConfig);
    const modelHubPublicModelsCallMock = vi.mocked(networking.modelHubPublicModelsCall);

    getUiConfigMock.mockResolvedValue({
      server_root_path: "/",
      proxy_base_url: "http://localhost:4000",
      auto_redirect_to_sso: false,
      admin_ui_disabled: false,
      sso_configured: false,
    });
    modelHubPublicModelsCallMock.mockResolvedValue([]);
    vi.mocked(networking.getUiSettings).mockResolvedValue({
      values: {},
    });
    mockUseUISettings.mockReturnValue({
      data: { values: {} },
      isLoading: false,
    });

    renderWithProviders(<ModelHubTable accessToken={null} publicPage={true} premiumUser={false} userRole={null} />);

    await waitFor(() => {
      expect(getUiConfigMock).toHaveBeenCalled();
      expect(modelHubPublicModelsCallMock).toHaveBeenCalled();
    });

    const getUiConfigCallOrder = getUiConfigMock.mock.invocationCallOrder[0];
    const modelHubPublicModelsCallOrder = modelHubPublicModelsCallMock.mock.invocationCallOrder[0];

    expect(getUiConfigCallOrder).toBeLessThan(modelHubPublicModelsCallOrder);
  });

  describe("hub tabs", () => {
    const renderHub = async (agents: object[] = []) => {
      vi.mocked(networking.modelHubCall).mockResolvedValue({
        data: [{ model_group: "claude-opus-4-8", providers: ["anthropic"], mode: "chat" }],
      });
      vi.mocked(networking.getConfigFieldSetting).mockResolvedValue({ field_value: false });
      vi.mocked(networking.getAgentsList).mockResolvedValue({ agents });
      vi.mocked(networking.fetchMCPServers).mockResolvedValue([]);
      vi.mocked(networking.getUiSettings).mockResolvedValue({ values: {} });
      mockUseUISettings.mockReturnValue({ data: { values: {} }, isLoading: false });

      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderWithProviders(
        <ModelHubTable accessToken="test-token" publicPage={false} premiumUser={false} userRole="Admin" />,
        { onUrlUpdate },
      );
      return { user, onUrlUpdate, search: await screen.findByPlaceholderText("Search model names...") };
    };
    const urlWritten = (onUrlUpdate: Mock<OnUrlUpdateFunction>, key: string, value: string) =>
      waitFor(() => expect(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.get(key)).toBe(value));

    it("keeps the model filter typed on the Model Hub tab after visiting another hub", async () => {
      const { user, onUrlUpdate, search } = await renderHub();

      fireEvent.change(search, { target: { value: "opus" } });
      await urlWritten(onUrlUpdate, "q", "opus");
      await user.click(screen.getByRole("tab", { name: "Agent Hub" }));
      await user.click(screen.getByRole("tab", { name: "Model Hub" }));

      expect(await screen.findByPlaceholderText("Search model names...")).toHaveValue("opus");
    });

    it("filters the Agent Hub table by name or description and shows the no-match state", async () => {
      const { user, onUrlUpdate } = await renderHub([
        {
          agent_id: "a1",
          agent_card_params: { name: "Billing Router", description: "routes billing questions" },
          litellm_params: { is_public: false },
        },
        {
          agent_id: "a2",
          agent_card_params: { name: "Support Bot", description: "handles support tickets" },
          litellm_params: { is_public: false },
        },
      ]);
      const agentCount = (expected: string) =>
        screen.getByText((_, el) => el?.tagName === "P" && el.textContent === expected);

      await user.click(screen.getByRole("tab", { name: "Agent Hub" }));
      expect(await screen.findByText("Billing Router")).toBeInTheDocument();

      const search = screen.getByPlaceholderText("Search agent names or descriptions...");
      fireEvent.change(search, { target: { value: "support tickets" } });
      await urlWritten(onUrlUpdate, "agent_q", "support tickets");
      await waitFor(() => expect(screen.queryByText("Billing Router")).not.toBeInTheDocument());
      expect(screen.getByText("Support Bot")).toBeInTheDocument();
      expect(agentCount("Showing 1 of 2 agents")).toBeInTheDocument();

      fireEvent.change(search, { target: { value: "zzzz" } });
      expect(await screen.findByText("No matching agents")).toBeInTheDocument();
      expect(agentCount("Showing 0 of 2 agents")).toBeInTheDocument();
    });

    it("renders the hub strip as underlined tabs rather than a segmented pill", async () => {
      await renderHub();

      expect(screen.getByRole("tablist")).toHaveAttribute("data-variant", "line");
    });
  });

  describe("URL state", () => {
    const hubModel = (model_group: string, provider: string, mode: string, extra: object = {}) => ({
      model_group,
      providers: [provider],
      mode,
      supports_vision: false,
      ...extra,
    });
    const MODELS = [
      hubModel("alpha-chat", "anthropic", "chat", { supports_vision: true }),
      hubModel("beta-embed", "openai", "embedding"),
    ];
    const agentEntry = (agent_id: string, name: string, description: string, is_public: boolean) => ({
      agent_id,
      agent_card_params: { name, description, version: "1.0.0" },
      litellm_params: { is_public },
    });
    const AGENTS = [
      agentEntry("a1", "Billing Router", "routes billing questions", true),
      agentEntry("a2", "Support Bot", "handles support tickets", false),
    ];
    const mcpServer = (server_id: string, server_name: string, created_by: string) => ({
      server_id,
      server_name,
      transport: "http",
      auth_type: "none",
      status: "active",
      created_at: "2026-01-01T00:00:00Z",
      updated_at: "2026-01-01T00:00:00Z",
      created_by,
      updated_by: created_by,
    });
    const MCP_SERVERS = [mcpServer("srv-1", "alpha-search", "zed"), mcpServer("srv-2", "beta-tickets", "amy")];
    const padded = (prefix: string, index: number) => `${prefix}-${String(index).padStart(2, "0")}`;
    const MANY_MODELS = Array.from({ length: 60 }, (_, index) =>
      hubModel(padded("model", index), index % 2 ? "openai" : "anthropic", "chat"),
    );
    const MANY_AGENTS = Array.from({ length: 60 }, (_, index) =>
      agentEntry(padded("id", index), padded("agent", index), "generated", false),
    );
    const MANY_MCP_SERVERS = Array.from({ length: 60 }, (_, index) =>
      mcpServer(padded("srv", index), padded("server", index), "zed"),
    );

    interface HubFixtures {
      models?: object[];
      agents?: object[];
      servers?: object[];
    }

    const renderUrlHub = async (searchParams = "", fixtures: HubFixtures = {}) => {
      const { models = MODELS, agents = AGENTS, servers = MCP_SERVERS } = fixtures;
      vi.mocked(networking.modelHubCall).mockResolvedValue({ data: models });
      vi.mocked(networking.getConfigFieldSetting).mockResolvedValue({ field_value: false });
      vi.mocked(networking.getAgentsList).mockResolvedValue({ agents });
      vi.mocked(networking.fetchMCPServers).mockResolvedValue(servers);
      mockUseUISettings.mockReturnValue({ data: { values: {} }, isLoading: false });

      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderWithProviders(
        <ModelHubTable accessToken="test-token" publicPage={false} premiumUser={false} userRole="Admin" />,
        { searchParams, onUrlUpdate },
      );
      await waitFor(() => expect(screen.queryByText("Loading models…")).not.toBeInTheDocument());
      await screen.findByText(new RegExp(`^Showing ${servers.length} MCP servers?$`));
      await screen.findByText(new RegExp(`of ${agents.length} agents`));
      return { user: userEvent.setup(), onUrlUpdate };
    };

    const lastUrl = (onUrlUpdate: Mock<OnUrlUpdateFunction>) => {
      const update = onUrlUpdate.mock.calls.at(-1)?.[0];
      if (!update) throw new Error("no URL update was emitted");
      return update;
    };

    const activePanel = () => screen.getByRole("tabpanel");
    const pageLabel = () => within(activePanel()).getByTestId("pagination-page");
    const nextPage = (user: ReturnType<typeof userEvent.setup>) =>
      user.click(within(activePanel()).getByTestId("pagination-next"));
    const rowNames = (names: RegExp) =>
      within(activePanel())
        .queryAllByRole("button", { name: names })
        .map((button) => button.textContent);
    const MODEL_ROWS = /^(alpha-chat|beta-embed)$/;
    const AGENT_ROWS = /^(Billing Router|Support Bot)$/;
    const MCP_ROWS = /^(alpha-search|beta-tickets)$/;

    it("opens the hub tab named by ?tab=", async () => {
      await renderUrlHub("?tab=mcp");

      expect(screen.getByRole("tab", { name: "MCP Hub" })).toHaveAttribute("aria-selected", "true");
      expect(rowNames(MCP_ROWS)).toEqual(["alpha-search", "beta-tickets"]);
    });

    it("writes the chosen hub tab to ?tab= and drops it for the Model Hub", async () => {
      const { user, onUrlUpdate } = await renderUrlHub();

      await user.click(screen.getByRole("tab", { name: "Skill Hub" }));
      await waitFor(() => expect(lastUrl(onUrlUpdate).searchParams.get("tab")).toBe("skills"));

      await user.click(screen.getByRole("tab", { name: "Model Hub" }));
      await waitFor(() => expect(lastUrl(onUrlUpdate).searchParams.has("tab")).toBe(false));
    });

    it("filters the models by the q, provider and mode in the URL", async () => {
      await renderUrlHub("?q=beta&provider=openai&mode=embedding");

      expect(screen.getByPlaceholderText("Search model names...")).toHaveValue("beta");
      expect(screen.getByDisplayValue("openai")).toBeInTheDocument();
      expect(screen.getByDisplayValue("embedding")).toBeInTheDocument();
      expect(rowNames(MODEL_ROWS)).toEqual(["beta-embed"]);
    });

    it("filters the models by the feature in the URL", async () => {
      await renderUrlHub("?feature=Vision");

      expect(screen.getByDisplayValue("Vision")).toBeInTheDocument();
      expect(rowNames(MODEL_ROWS)).toEqual(["alpha-chat"]);
    });

    it("writes the model filters to the URL and Clear Filters drops every one of them", async () => {
      const { user, onUrlUpdate } = await renderUrlHub();

      fireEvent.change(screen.getByPlaceholderText("Search model names..."), { target: { value: "a" } });
      await waitFor(() => expect(lastUrl(onUrlUpdate).searchParams.get("q")).toBe("a"));
      fireEvent.change(screen.getByDisplayValue("All Providers"), { target: { value: "anthropic" } });
      await waitFor(() => expect(lastUrl(onUrlUpdate).searchParams.get("provider")).toBe("anthropic"));
      fireEvent.change(screen.getByDisplayValue("All Modes"), { target: { value: "chat" } });
      await waitFor(() => expect(lastUrl(onUrlUpdate).searchParams.get("mode")).toBe("chat"));
      fireEvent.change(screen.getByDisplayValue("All Features"), { target: { value: "Vision" } });
      await waitFor(() => expect(lastUrl(onUrlUpdate).searchParams.get("feature")).toBe("Vision"));
      expect(rowNames(MODEL_ROWS)).toEqual(["alpha-chat"]);

      await user.click(screen.getByRole("button", { name: "Clear Filters" }));

      await waitFor(() => expect(lastUrl(onUrlUpdate).queryString).toBe(""));
      expect(rowNames(MODEL_ROWS)).toEqual(["alpha-chat", "beta-embed"]);
    });

    it("reads the model table page from models_page and writes page changes back", async () => {
      const { user, onUrlUpdate } = await renderUrlHub("?models_page=2", { models: MANY_MODELS });
      expect(pageLabel()).toHaveTextContent("Page 2 of 3");

      await nextPage(user);

      await waitFor(() => expect(lastUrl(onUrlUpdate).searchParams.get("models_page")).toBe("3"));
      expect(pageLabel()).toHaveTextContent("Page 3 of 3");
    });

    it("returns the model table to its first page when a model filter changes", async () => {
      const { onUrlUpdate } = await renderUrlHub("?models_page=2", { models: MANY_MODELS });
      expect(pageLabel()).toHaveTextContent("Page 2 of 3");

      fireEvent.change(screen.getByDisplayValue("All Providers"), { target: { value: "openai" } });

      await waitFor(() => expect(lastUrl(onUrlUpdate).searchParams.get("provider")).toBe("openai"));
      expect(lastUrl(onUrlUpdate).searchParams.has("models_page")).toBe(false);
      expect(pageLabel()).toHaveTextContent("Page 1 of 2");
    });

    it("sorts the models table from models_ keys and writes header clicks back", async () => {
      const { user, onUrlUpdate } = await renderUrlHub("?models_sort_order=desc");
      expect(rowNames(MODEL_ROWS)).toEqual(["beta-embed", "alpha-chat"]);

      await user.click(within(activePanel()).getByTestId("sort-header-mode"));

      await waitFor(() => expect(lastUrl(onUrlUpdate).searchParams.get("models_sort_by")).toBe("mode"));
      expect(rowNames(MODEL_ROWS)).toEqual(["alpha-chat", "beta-embed"]);
    });

    it("reads the agent search from agent_q", async () => {
      await renderUrlHub("?tab=agents&agent_q=support");

      expect(screen.getByPlaceholderText("Search agent names or descriptions...")).toHaveValue("support");
      expect(rowNames(AGENT_ROWS)).toEqual(["Support Bot"]);
    });

    it("writes the agent search to agent_q and header sorts to agents_ keys", async () => {
      const { user, onUrlUpdate } = await renderUrlHub("?tab=agents&agents_sort_order=desc");
      expect(rowNames(AGENT_ROWS)).toEqual(["Support Bot", "Billing Router"]);

      fireEvent.change(screen.getByPlaceholderText("Search agent names or descriptions..."), {
        target: { value: "o" },
      });
      await waitFor(() => expect(lastUrl(onUrlUpdate).searchParams.get("agent_q")).toBe("o"));

      await user.click(within(activePanel()).getByTestId("sort-header-name"));
      await waitFor(() => expect(lastUrl(onUrlUpdate).searchParams.has("agents_sort_order")).toBe(false));
      expect(lastUrl(onUrlUpdate).searchParams.get("agent_q")).toBe("o");
      expect(rowNames(AGENT_ROWS)).toEqual(["Billing Router", "Support Bot"]);
    });

    it("sorts the agents table from agents_sort_by and writes header clicks back", async () => {
      const { user, onUrlUpdate } = await renderUrlHub("?tab=agents&agents_sort_by=description");
      expect(rowNames(AGENT_ROWS)).toEqual(["Support Bot", "Billing Router"]);

      await user.click(within(activePanel()).getByTestId("sort-header-version"));

      await waitFor(() => expect(lastUrl(onUrlUpdate).searchParams.get("agents_sort_by")).toBe("version"));
      expect(lastUrl(onUrlUpdate).searchParams.has("agents_sort_order")).toBe(false);
    });

    it("reads the agents table page from agents_page and writes page changes back", async () => {
      const { user, onUrlUpdate } = await renderUrlHub("?tab=agents&agents_page=2", { agents: MANY_AGENTS });
      expect(pageLabel()).toHaveTextContent("Page 2 of 3");
      expect(within(activePanel()).getByRole("button", { name: "agent-25" })).toBeInTheDocument();

      await nextPage(user);

      await waitFor(() => expect(lastUrl(onUrlUpdate).searchParams.get("agents_page")).toBe("3"));
      expect(pageLabel()).toHaveTextContent("Page 3 of 3");
      expect(within(activePanel()).getByRole("button", { name: "agent-50" })).toBeInTheDocument();
    });

    it("reads the MCP table page from mcp_page and writes page changes back", async () => {
      const { user, onUrlUpdate } = await renderUrlHub("?tab=mcp&mcp_page=2", { servers: MANY_MCP_SERVERS });
      expect(pageLabel()).toHaveTextContent("Page 2 of 3");
      expect(within(activePanel()).getByRole("button", { name: "server-25" })).toBeInTheDocument();

      await nextPage(user);

      await waitFor(() => expect(lastUrl(onUrlUpdate).searchParams.get("mcp_page")).toBe("3"));
      expect(pageLabel()).toHaveTextContent("Page 3 of 3");
      expect(within(activePanel()).getByRole("button", { name: "server-50" })).toBeInTheDocument();
    });

    it("sorts the MCP table from mcp_ keys and writes header clicks back", async () => {
      const { user, onUrlUpdate } = await renderUrlHub("?tab=mcp&mcp_sort_order=desc");
      expect(rowNames(MCP_ROWS)).toEqual(["beta-tickets", "alpha-search"]);

      await user.click(within(activePanel()).getByTestId("sort-header-created_by"));

      await waitFor(() => expect(lastUrl(onUrlUpdate).searchParams.get("mcp_sort_by")).toBe("created_by"));
      expect(rowNames(MCP_ROWS)).toEqual(["beta-tickets", "alpha-search"]);
    });

    it("opens the model named by ?model=", async () => {
      await renderUrlHub("?model=beta-embed");

      const dialog = await screen.findByRole("dialog");
      expect(within(dialog).getByRole("heading", { name: "beta-embed" })).toBeInTheDocument();
      expect(dialog).toHaveTextContent("Model Overview");
    });

    it("opens the agent whose id is in ?agent=", async () => {
      await renderUrlHub("?tab=agents&agent=a2");

      const dialog = await screen.findByRole("dialog");
      expect(within(dialog).getByRole("heading", { name: "Support Bot" })).toBeInTheDocument();
      expect(dialog).toHaveTextContent("Agent Overview");
    });

    it("opens the agent named in ?agent= by a link from the public hub", async () => {
      await renderUrlHub("?tab=agents&agent=Support%20Bot");

      const dialog = await screen.findByRole("dialog");
      expect(within(dialog).getByRole("heading", { name: "Support Bot" })).toBeInTheDocument();
    });

    it("opens the MCP server whose id is in ?mcp=", async () => {
      await renderUrlHub("?tab=mcp&mcp=srv-2");

      const dialog = await screen.findByRole("dialog");
      expect(within(dialog).getByRole("heading", { name: "beta-tickets" })).toBeInTheDocument();
      expect(dialog).toHaveTextContent("Server Overview");
    });

    it("leaves ?model= to the public hub on the public route", async () => {
      vi.mocked(networking.modelHubCall).mockResolvedValue({ data: MODELS });
      vi.mocked(networking.getConfigFieldSetting).mockResolvedValue({ field_value: false });
      mockUseUISettings.mockReturnValue({ data: { values: {} }, isLoading: false });

      renderWithProviders(
        <ModelHubTable accessToken="test-token" publicPage={true} premiumUser={false} userRole={null} />,
        { searchParams: "?model=alpha-chat" },
      );

      expect(await screen.findByText("Public Model Hub not enabled.")).toBeInTheDocument();
      await waitFor(() => expect(networking.getConfigFieldSetting).toHaveBeenCalled());
      await act(async () => {});
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    });

    it("pushes the clicked model into ?model= and clears it when the dialog closes", async () => {
      const { user, onUrlUpdate } = await renderUrlHub();

      await user.click(within(activePanel()).getByRole("button", { name: "alpha-chat" }));

      await waitFor(() => expect(lastUrl(onUrlUpdate).searchParams.get("model")).toBe("alpha-chat"));
      expect(lastUrl(onUrlUpdate).options.history).toBe("push");
      const dialog = await screen.findByRole("dialog");
      expect(within(dialog).getByRole("heading", { name: "alpha-chat" })).toBeInTheDocument();

      await user.click(within(dialog).getByRole("button", { name: /close/i }));

      await waitFor(() => expect(lastUrl(onUrlUpdate).searchParams.has("model")).toBe(false));
      await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    });

    it("pushes the clicked agent id into ?agent= and the clicked server id into ?mcp=", async () => {
      const { user, onUrlUpdate } = await renderUrlHub("?tab=agents");

      await user.click(within(activePanel()).getByRole("button", { name: "Billing Router" }));
      await waitFor(() => expect(lastUrl(onUrlUpdate).searchParams.get("agent")).toBe("a1"));
      expect(lastUrl(onUrlUpdate).options.history).toBe("push");
      await user.click(within(await screen.findByRole("dialog")).getByRole("button", { name: /close/i }));
      await waitFor(() => expect(lastUrl(onUrlUpdate).searchParams.has("agent")).toBe(false));

      await user.click(screen.getByRole("tab", { name: "MCP Hub" }));
      await waitFor(() => expect(lastUrl(onUrlUpdate).searchParams.get("tab")).toBe("mcp"));
      await user.click(within(activePanel()).getByRole("button", { name: "beta-tickets" }));
      await waitFor(() => expect(lastUrl(onUrlUpdate).searchParams.get("mcp")).toBe("srv-2"));
      expect(lastUrl(onUrlUpdate).options.history).toBe("push");
    });

    it("keeps each agent's public flag when the list is refreshed after publishing", async () => {
      const { user } = await renderUrlHub("?tab=agents");
      const publicCell = (name: string) =>
        within(within(activePanel()).getByRole("row", { name: new RegExp(`^${name}`) })).getByText(/^(Yes|No)$/);
      expect(publicCell("Billing Router")).toHaveTextContent("Yes");
      expect(publicCell("Support Bot")).toHaveTextContent("No");
      vi.mocked(networking.getAgentsList).mockResolvedValue({
        agents: [
          agentEntry("a1", "Billing Router", "routes billing questions", false),
          agentEntry("a2", "Support Bot", "handles support tickets", true),
        ],
      });

      await user.click(screen.getByRole("button", { name: "Select Agents to Make Public" }));
      await user.click(screen.getByRole("button", { name: "Confirm agent publish" }));

      await waitFor(() => expect(publicCell("Support Bot")).toHaveTextContent("Yes"));
      expect(publicCell("Billing Router")).toHaveTextContent("No");
    });
  });

  describe("authentication redirect behavior", () => {
    // Test cases where requireAuth is true - should redirect on invalid tokens
    testAuthRedirect(
      true,
      null,
      false,
      true,
      "should redirect to login when requireAuth is true and there is no token",
    );

    testAuthRedirect(
      true,
      "expired-token",
      false,
      true,
      "should redirect to login when requireAuth is true and token is expired",
    );

    testAuthRedirect(
      true,
      "malformed-token",
      false,
      true,
      "should redirect to login when requireAuth is true and token is malformed",
    );

    // Test cases where requireAuth is false - should NOT redirect regardless of token state
    testAuthRedirect(false, null, false, false, "should not redirect when requireAuth is false and there is no token");

    testAuthRedirect(
      false,
      "expired-token",
      false,
      false,
      "should not redirect when requireAuth is false and token is expired",
    );

    testAuthRedirect(
      false,
      "malformed-token",
      false,
      false,
      "should not redirect when requireAuth is false and token is malformed",
    );
  });
});
