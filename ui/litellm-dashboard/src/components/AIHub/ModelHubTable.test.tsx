import * as networking from "@/components/networking";
import userEvent from "@testing-library/user-event";
import { act } from "@testing-library/react";
import type { MCPServerData } from "./MCPHubTableColumns";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders, screen, testQueryClient, waitFor, within } from "../../../tests/test-utils";
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
  getGlobalLitellmHeaderName: vi.fn(() => "X-Litellm-Key"),
  getProxyBaseUrl: vi.fn(() => "http://localhost:4000"),
  getAgentsList: vi.fn(),
  fetchMCPServers: vi.fn(),
  makeMCPPublicCall: vi.fn(),
  getUiSettings: vi.fn(),
  getClaudeCodePluginsList: vi.fn(() => Promise.resolve({ plugins: [] })),
  modelCostMap: vi.fn(() => Promise.resolve({})),
  getProviderCreateMetadata: vi.fn(() => Promise.resolve([])),
}));

const GARDEN_COST_MAP = {
  "typesafe/jev-latest": { litellm_provider: "typesafe", mode: "evaluation" },
  "typesafe/decisions/jev-latest": { litellm_provider: "typesafe", mode: "evaluation" },
  "cloudflare/clef": { litellm_provider: "cloudflare", mode: "evaluation" },
  "claude-opus-4-8": { litellm_provider: "anthropic", mode: "chat" },
};

const ADDABLE_PROVIDER_FIELDS = [
  { provider: "TypeSafe", provider_display_name: "TypeSafe", litellm_provider: "typesafe", credential_fields: [] },
];

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    replace: mockRouterReplace,
  }),
}));

vi.mock("@/components/public_model_hub", () => ({
  default: () => <div>Public Model Hub</div>,
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
    testQueryClient.clear();
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

      renderWithProviders(
        <ModelHubTable
          accessToken={null}
          publicPage={true}
          premiumUser={false}
          userRole={null}
          canOpenPlayground={false}
          canAddModels={false}
        />,
      );

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
      <ModelHubTable
        accessToken="test-token"
        publicPage={false}
        premiumUser={false}
        userRole={null}
        canOpenPlayground
        canAddModels
      />,
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

    renderWithProviders(
      <ModelHubTable
        accessToken={null}
        publicPage={false}
        premiumUser={false}
        userRole={null}
        canOpenPlayground
        canAddModels
      />,
    );

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

    renderWithProviders(
      <ModelHubTable
        accessToken={null}
        publicPage={true}
        premiumUser={false}
        userRole={null}
        canOpenPlayground={false}
        canAddModels={false}
      />,
    );

    await waitFor(() => {
      expect(getUiConfigMock).toHaveBeenCalled();
      expect(modelHubPublicModelsCallMock).toHaveBeenCalled();
    });

    const getUiConfigCallOrder = getUiConfigMock.mock.invocationCallOrder[0];
    const modelHubPublicModelsCallOrder = modelHubPublicModelsCallMock.mock.invocationCallOrder[0];

    expect(getUiConfigCallOrder).toBeLessThan(modelHubPublicModelsCallOrder);
  });

  describe("model details usage example", () => {
    const openDetails = async (
      model: { model_group: string; providers: string[]; mode: string },
      canOpenPlayground = true,
    ) => {
      vi.mocked(networking.modelHubCall).mockResolvedValue({ data: [model] });
      vi.mocked(networking.getConfigFieldSetting).mockResolvedValue({ field_value: false });
      vi.mocked(networking.getAgentsList).mockResolvedValue({ agents: [] });
      vi.mocked(networking.fetchMCPServers).mockResolvedValue([]);
      vi.mocked(networking.getUiSettings).mockResolvedValue({ values: {} });
      mockUseUISettings.mockReturnValue({ data: { values: {} }, isLoading: false });
      const user = userEvent.setup();
      renderWithProviders(
        <ModelHubTable
          accessToken="test-token"
          publicPage={false}
          premiumUser={false}
          userRole="Admin"
          canOpenPlayground={canOpenPlayground}
          canAddModels
        />,
      );
      await user.click(await screen.findByRole("button", { name: model.model_group }));
      return screen.findByRole("dialog");
    };

    it("shows a decision model how to call it on /v1/systemone, with the docs and the playground", async () => {
      const dialog = await openDetails({ model_group: "jev-latest", providers: ["typesafe"], mode: "evaluation" });

      expect(dialog).toHaveTextContent('"http://localhost:4000/v1/systemone"');
      expect(dialog).toHaveTextContent('headers={"X-Litellm-Key": "Bearer your_api_key"}');
      expect(dialog).not.toHaveTextContent("chat.completions");
      expect(screen.getByRole("link", { name: "How to call decision models" })).toHaveAttribute(
        "href",
        "https://docs.litellm.ai/docs/decisions",
      );
      expect(screen.getByRole("link", { name: "Try it in the Playground" })).toHaveAttribute(
        "href",
        "/ui/playground?tab=system-one",
      );
    });

    it("does not link a view-only session to the Playground it cannot open", async () => {
      const dialog = await openDetails(
        { model_group: "jev-latest", providers: ["typesafe"], mode: "evaluation" },
        false,
      );

      expect(dialog).toHaveTextContent('"http://localhost:4000/v1/systemone"');
      expect(screen.getByRole("link", { name: "How to call decision models" })).toBeInTheDocument();
      expect(screen.queryByRole("link", { name: "Try it in the Playground" })).not.toBeInTheDocument();
    });

    it("keeps the chat completions example for a chat model", async () => {
      const dialog = await openDetails({ model_group: "claude-opus-4-8", providers: ["anthropic"], mode: "chat" });

      expect(dialog).toHaveTextContent("client.chat.completions.create(");
      expect(screen.queryByRole("link", { name: "How to call decision models" })).not.toBeInTheDocument();
    });
  });

  describe("decision model garden", () => {
    const renderHub = async (
      models: { model_group: string; providers: string[]; mode: string }[],
      { userRole = "Admin", canAddModels = true }: { userRole?: string; canAddModels?: boolean } = {},
    ) => {
      vi.mocked(networking.modelHubCall).mockResolvedValue({ data: models });
      vi.mocked(networking.modelCostMap).mockResolvedValue(GARDEN_COST_MAP);
      vi.mocked(networking.getProviderCreateMetadata).mockResolvedValue(ADDABLE_PROVIDER_FIELDS);
      vi.mocked(networking.getConfigFieldSetting).mockResolvedValue({ field_value: false });
      vi.mocked(networking.getAgentsList).mockResolvedValue({ agents: [] });
      vi.mocked(networking.fetchMCPServers).mockResolvedValue([]);
      vi.mocked(networking.getUiSettings).mockResolvedValue({ values: {} });
      mockUseUISettings.mockReturnValue({ data: { values: {} }, isLoading: false });
      const user = userEvent.setup();
      renderWithProviders(
        <ModelHubTable
          accessToken="test-token"
          publicPage={false}
          premiumUser={false}
          userRole={userRole}
          canOpenPlayground
          canAddModels={canAddModels}
        />,
      );
      await screen.findByText(/Showing \d+ of \d+ models/);
      return { user, modeSelect: () => screen.getByDisplayValue("All Modes") };
    };

    const chatOnly = [{ model_group: "claude-opus-4-8", providers: ["anthropic"], mode: "chat" }];

    it("lists the decision models from the cost map when the proxy has none configured", async () => {
      await renderHub(chatOnly);

      const garden = await screen.findByTestId("decision-model-garden");
      expect(garden).toHaveTextContent("2 decision models across 2 providers");
      expect(within(garden).getByTestId("decision-model-card-typesafe/jev-latest")).toHaveTextContent("TypeSafe");
      expect(within(garden).getByTestId("decision-model-card-cloudflare/clef")).toHaveTextContent("Cloudflare");
      expect(within(garden).queryByTestId("decision-model-card-typesafe/decisions/jev-latest")).not.toBeInTheDocument();
      expect(within(garden).queryByText("claude-opus-4-8")).not.toBeInTheDocument();
    });

    it("links each card to the decisions docs and to Add Model prefilled with that provider and model", async () => {
      await renderHub(chatOnly);

      const card = within(await screen.findByTestId("decision-model-garden")).getByTestId(
        "decision-model-card-typesafe/jev-latest",
      );
      expect(within(card).getByRole("link", { name: "/v1/decisions docs" })).toHaveAttribute(
        "href",
        "https://docs.litellm.ai/docs/decisions",
      );
      expect(await within(card).findByRole("link", { name: "Add model" })).toHaveAttribute(
        "href",
        expect.stringMatching(/\/models-and-endpoints\?tab=add&provider=typesafe&litellm_model=typesafe%2Fjev-latest$/),
      );
    });

    it("offers a decisions mode option that keeps the garden and hides the chat rows", async () => {
      const { user, modeSelect } = await renderHub(chatOnly);
      await screen.findByTestId("decision-model-garden");

      expect(within(modeSelect()).queryByRole("option", { name: "evaluation" })).not.toBeInTheDocument();
      await user.selectOptions(modeSelect(), "decisions");

      expect(screen.getByTestId("decision-model-garden")).toBeInTheDocument();
      expect(screen.queryByRole("button", { name: "claude-opus-4-8" })).not.toBeInTheDocument();
      expect(screen.getByText("Showing 0 of 1 models")).toBeInTheDocument();
    });

    it("hides the garden while a non-decision mode is selected", async () => {
      const { user, modeSelect } = await renderHub(chatOnly);
      await screen.findByTestId("decision-model-garden");

      await user.selectOptions(modeSelect(), "chat");
      expect(screen.queryByTestId("decision-model-garden")).not.toBeInTheDocument();

      await user.click(screen.getByRole("button", { name: "Clear Filters" }));
      expect(await screen.findByTestId("decision-model-garden")).toBeInTheDocument();
    });

    it("labels a configured evaluation model as decisions in the table and under the decisions filter", async () => {
      const { user, modeSelect } = await renderHub([
        ...chatOnly,
        { model_group: "jev-latest", providers: ["typesafe"], mode: "evaluation" },
      ]);

      const row = screen.getByRole("button", { name: "jev-latest" }).closest("tr");
      expect(row).toHaveTextContent("decisions");
      expect(row).not.toHaveTextContent("evaluation");

      await user.selectOptions(modeSelect(), "decisions");
      expect(screen.getByText("Showing 1 of 2 models")).toBeInTheDocument();
      expect(screen.getByRole("button", { name: "jev-latest" })).toBeInTheDocument();
    });

    it("drops the Add Model link for an admin session that cannot add models", async () => {
      await renderHub(chatOnly, { canAddModels: false });

      const garden = await screen.findByTestId("decision-model-garden");
      expect(within(garden).queryByRole("link", { name: "Add model" })).not.toBeInTheDocument();
      expect(within(garden).getAllByRole("link", { name: "/v1/decisions docs" })).toHaveLength(2);
    });

    it("drops the Add Model link on a card whose provider the Add Model form cannot create", async () => {
      await renderHub(chatOnly);

      const garden = await screen.findByTestId("decision-model-garden");
      const typesafeCard = within(garden).getByTestId("decision-model-card-typesafe/jev-latest");
      const cloudflareCard = within(garden).getByTestId("decision-model-card-cloudflare/clef");
      expect(await within(typesafeCard).findByRole("link", { name: "Add model" })).toBeInTheDocument();
      expect(within(cloudflareCard).queryByRole("link", { name: "Add model" })).not.toBeInTheDocument();
      expect(within(cloudflareCard).getByRole("link", { name: "/v1/decisions docs" })).toBeInTheDocument();
    });

    it("sorts the Mode column by the label it shows, placing decisions between chat and embedding", async () => {
      const { user } = await renderHub([
        { model_group: "text-embedding-4", providers: ["openai"], mode: "embedding" },
        { model_group: "jev-latest", providers: ["typesafe"], mode: "evaluation" },
        ...chatOnly,
      ]);

      await user.click(screen.getByTestId("sort-header-mode"));

      const order = screen
        .getAllByRole("button", { name: /^(claude-opus-4-8|jev-latest|text-embedding-4)$/ })
        .map((button) => button.textContent);
      expect(order).toEqual(["claude-opus-4-8", "jev-latest", "text-embedding-4"]);
    });
  });

  describe("hub tabs", () => {
    const renderHub = async (agents: object[] = [], mcpServers: Promise<MCPServerData[]> = Promise.resolve([])) => {
      vi.mocked(networking.modelHubCall).mockResolvedValue({
        data: [{ model_group: "claude-opus-4-8", providers: ["anthropic"], mode: "chat" }],
      });
      vi.mocked(networking.getConfigFieldSetting).mockResolvedValue({ field_value: false });
      vi.mocked(networking.getAgentsList).mockResolvedValue({ agents });
      vi.mocked(networking.fetchMCPServers).mockReturnValue(mcpServers);
      vi.mocked(networking.getUiSettings).mockResolvedValue({ values: {} });
      mockUseUISettings.mockReturnValue({ data: { values: {} }, isLoading: false });

      const user = userEvent.setup();
      renderWithProviders(
        <ModelHubTable
          accessToken="test-token"
          publicPage={false}
          premiumUser={false}
          userRole="Admin"
          canOpenPlayground
          canAddModels
        />,
      );
      return { user, search: await screen.findByPlaceholderText("Search model names...") };
    };

    it("requires a fresh MCP publication list before and after saving", async () => {
      const servers = Promise.withResolvers<MCPServerData[]>();
      const { user } = await renderHub([], servers.promise);
      await user.click(screen.getByRole("tab", { name: "MCP Hub" }));

      const manageVisibility = screen.getByRole("button", { name: "Manage MCP Hub Visibility" });
      expect(manageVisibility).toBeDisabled();
      await act(async () => servers.resolve([]));
      expect(manageVisibility).toBeEnabled();

      const refresh = Promise.withResolvers<MCPServerData[]>();
      vi.mocked(networking.makeMCPPublicCall).mockResolvedValueOnce({});
      vi.mocked(networking.fetchMCPServers).mockReturnValueOnce(refresh.promise);
      await user.click(manageVisibility);
      await user.click(screen.getByRole("button", { name: "Next" }));
      await user.click(screen.getByRole("button", { name: "Save Publication List" }));

      expect(networking.makeMCPPublicCall).toHaveBeenCalledWith("test-token", []);
      expect(manageVisibility).toBeDisabled();
      await act(async () => refresh.reject(new Error("Unable to reload the publication list")));
      expect(manageVisibility).toBeDisabled();
    });

    it("keeps the model filter typed on the Model Hub tab after visiting another hub", async () => {
      const { user, search } = await renderHub();

      await user.type(search, "opus");
      await user.click(screen.getByRole("tab", { name: "Agent Hub" }));
      await user.click(screen.getByRole("tab", { name: "Model Hub" }));

      expect(await screen.findByPlaceholderText("Search model names...")).toHaveValue("opus");
    });

    it("filters the Agent Hub table by name or description and shows the no-match state", async () => {
      const { user } = await renderHub([
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
      await user.type(search, "support tickets");
      expect(screen.queryByText("Billing Router")).not.toBeInTheDocument();
      expect(screen.getByText("Support Bot")).toBeInTheDocument();
      expect(agentCount("Showing 1 of 2 agents")).toBeInTheDocument();

      await user.clear(search);
      await user.type(search, "zzzz");
      expect(screen.getByText("No matching agents")).toBeInTheDocument();
      expect(agentCount("Showing 0 of 2 agents")).toBeInTheDocument();
    });

    it("renders the hub strip as underlined tabs rather than a segmented pill", async () => {
      await renderHub();

      expect(screen.getByRole("tablist")).toHaveAttribute("data-variant", "line");
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
