/* @vitest-environment jsdom */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import ModelsAndEndpointsPage from "./page";

vi.mock("./panels/AllModelsPanel", () => ({ default: () => <div data-testid="panel-all-models" /> }));
vi.mock("./panels/AddModelPanel", () => ({ default: () => <div data-testid="panel-add" /> }));
vi.mock("./panels/AutoRoutersTabPanel", () => ({ default: () => <div data-testid="panel-auto-routers" /> }));
vi.mock("./panels/LlmCredentialsPanel", () => ({ default: () => <div data-testid="panel-credentials" /> }));
vi.mock("./panels/PassThroughPanel", () => ({ default: () => <div data-testid="panel-pass-through" /> }));
vi.mock("./panels/HealthStatusPanel", () => ({ default: () => <div data-testid="panel-health" /> }));
vi.mock("./panels/ModelRetrySettingsPanel", () => ({ default: () => <div data-testid="panel-retry" /> }));
vi.mock("./panels/ModelGroupAliasPanel", () => ({ default: () => <div data-testid="panel-alias" /> }));
vi.mock("./panels/PriceDataPanel", () => ({ default: () => <div data-testid="panel-price" /> }));
vi.mock("./panels/AccessGroupBudgetsPanel", () => ({ default: () => <div data-testid="panel-budgets" /> }));

const detailState = { modelId: null as string | null, teamId: null as string | null };
vi.mock("./detailNavigation", () => ({
  useModelDetailRouting: () => ({ ...detailState, close: vi.fn(), openModel: vi.fn(), openTeam: vi.fn() }),
}));

vi.mock("@/components/molecules/cost_optimization_feedback_banner", () => ({ default: () => null }));
vi.mock("@/components/model_info_view", () => ({
  default: ({ modelId }: { modelId: string }) => <div data-testid="model-info">model:{modelId}</div>,
}));
vi.mock("@/components/team/TeamInfo", () => ({
  default: ({ teamId }: { teamId: string }) => <div data-testid="team-info">team:{teamId}</div>,
}));

const mockUseAuthorized = vi.fn();
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({ default: () => mockUseAuthorized() }));
vi.mock("@/app/(dashboard)/hooks/teams/useTeams", () => ({ useTeams: () => ({ data: [] }) }));
vi.mock("@/app/(dashboard)/hooks/uiSettings/useUISettings", () => ({
  useUISettings: () => ({ data: { values: {} } }),
}));
vi.mock("./useModelDashboardData", () => ({
  useModelDashboardData: () => ({ availableModelAccessGroups: [], allModelsOnProxy: [], availableModelGroups: [] }),
}));

const ADMIN = { accessToken: "at", token: "t", userRole: "Admin", userId: "u1", premiumUser: false, isViewOnly: false };
const NON_ADMIN = {
  accessToken: "at",
  token: "t",
  userRole: "Internal User",
  userId: "u1",
  premiumUser: false,
  isViewOnly: false,
};
// What useAuthorized returns for a proxy_admin_viewer session: effectiveSessionRole masquerades
// the role as "Admin" for read parity, and only isViewOnly tells the page it may not write.
const VIEW_ONLY_ADMIN = { ...ADMIN, isViewOnly: true };

const renderPage = () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  return render(
    <QueryClientProvider client={queryClient}>
      <ModelsAndEndpointsPage />
    </QueryClientProvider>,
  );
};

describe("ModelsAndEndpointsPage", () => {
  beforeEach(() => {
    detailState.modelId = null;
    detailState.teamId = null;
    mockUseAuthorized.mockReturnValue(ADMIN);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (global as any).ResizeObserver = class {
      observe() {}
      unobserve() {}
      disconnect() {}
    };
  });

  it("renders the admin tab bar and the All Models panel by default", () => {
    renderPage();
    expect(screen.getByRole("tab", { name: "All Models" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "LLM Credentials" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Health Status" })).toBeInTheDocument();
    expect(screen.getByTestId("panel-all-models")).toBeInTheDocument();
  });

  it("switches tabs in-memory, mounting only the active panel", async () => {
    const user = userEvent.setup();
    renderPage();
    await user.click(screen.getByRole("tab", { name: "Health Status" }));
    expect(screen.getByTestId("panel-health")).toBeInTheDocument();
    expect(screen.queryByTestId("panel-all-models")).not.toBeInTheDocument();
  });

  it("renders the model detail overlay from the ?model drill-in and hides the tabs", () => {
    detailState.modelId = "abc-123";
    renderPage();
    expect(screen.getByTestId("model-info")).toHaveTextContent("model:abc-123");
    expect(screen.queryByRole("tab", { name: "All Models" })).not.toBeInTheDocument();
  });

  it("renders the team detail overlay from the ?team drill-in", () => {
    detailState.teamId = "team-9";
    renderPage();
    expect(screen.getByTestId("team-info")).toHaveTextContent("team:team-9");
  });

  it("hides admin-only tabs for a non-admin user", () => {
    mockUseAuthorized.mockReturnValue(NON_ADMIN);
    renderPage();
    expect(screen.queryByRole("tab", { name: "LLM Credentials" })).not.toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: "Health Status" })).not.toBeInTheDocument();
  });

  it("keeps the full admin tab order for a real admin", () => {
    const { getAllByRole } = renderPage();
    expect(getAllByRole("tab").map((tab) => tab.textContent)).toEqual([
      "All Models",
      "Add Model",
      "Auto-Routers Beta",
      "LLM Credentials",
      "Pass-Through Endpoints",
      "Health Status",
      "Model Retry Settings",
      "Model Group Alias",
      "Model Access Group Budgets Beta",
      "Price Data Reload",
    ]);
  });

  it("hides the admin write-form tabs from a view-only admin, keeping the read views", () => {
    mockUseAuthorized.mockReturnValue(VIEW_ONLY_ADMIN);
    const { getByRole, queryByRole } = renderPage();
    expect(getByRole("tab", { name: "All Models" })).toBeInTheDocument();
    expect(getByRole("tab", { name: "Health Status" })).toBeInTheDocument();
    expect(queryByRole("tab", { name: "LLM Credentials" })).not.toBeInTheDocument();
    expect(queryByRole("tab", { name: "Pass-Through Endpoints" })).not.toBeInTheDocument();
    expect(queryByRole("tab", { name: "Model Retry Settings" })).not.toBeInTheDocument();
    expect(queryByRole("tab", { name: "Model Group Alias" })).not.toBeInTheDocument();
    expect(queryByRole("tab", { name: /Model Access Group Budgets/ })).not.toBeInTheDocument();
    expect(queryByRole("tab", { name: "Price Data Reload" })).not.toBeInTheDocument();
  });

  // Auto-routers are excluded from the All Models table, so this tab is their home: the only
  // place in the product to list, create, edit or delete one.
  describe("Auto-Routers tab", () => {
    it("sits third, after All Models and Add Model", () => {
      renderPage();

      const tabs = screen.getAllByRole("tab").map((tab) => tab.textContent);
      expect(tabs[0]).toContain("All Models");
      expect(tabs[1]).toBe("Add Model");
      expect(tabs[2]).toContain("Auto-Routers");
      // Badged Beta while the tab settles; BetaBadge renders the label text.
      expect(tabs[2]).toContain("Beta");
    });

    it("renders its panel when selected", async () => {
      const user = userEvent.setup();
      renderPage();

      await user.click(screen.getByRole("tab", { name: /Auto-Routers/ }));
      expect(screen.getByTestId("panel-auto-routers")).toBeInTheDocument();
    });

    it("is hidden from non-admins, who cannot write models", () => {
      mockUseAuthorized.mockReturnValue(NON_ADMIN);
      renderPage();

      expect(screen.queryByRole("tab", { name: /Auto-Routers/ })).not.toBeInTheDocument();
    });
  });
});
