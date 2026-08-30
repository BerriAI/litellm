/* @vitest-environment jsdom */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render } from "@testing-library/react";
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

const ADMIN = { accessToken: "at", token: "t", userRole: "Admin", userId: "u1", premiumUser: false };
const NON_ADMIN = { accessToken: "at", token: "t", userRole: "Internal User", userId: "u1", premiumUser: false };

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
    const { getByRole, getByTestId } = renderPage();
    expect(getByRole("tab", { name: "All Models" })).toBeInTheDocument();
    expect(getByRole("tab", { name: "LLM Credentials" })).toBeInTheDocument();
    expect(getByRole("tab", { name: "Health Status" })).toBeInTheDocument();
    expect(getByTestId("panel-all-models")).toBeInTheDocument();
  });

  it("switches tabs in-memory, mounting only the active panel", async () => {
    const user = userEvent.setup();
    const { getByRole, getByTestId, queryByTestId } = renderPage();
    await user.click(getByRole("tab", { name: "Health Status" }));
    expect(getByTestId("panel-health")).toBeInTheDocument();
    expect(queryByTestId("panel-all-models")).not.toBeInTheDocument();
  });

  it("renders the model detail overlay from the ?model drill-in and hides the tabs", () => {
    detailState.modelId = "abc-123";
    const { getByTestId, queryByRole } = renderPage();
    expect(getByTestId("model-info")).toHaveTextContent("model:abc-123");
    expect(queryByRole("tab", { name: "All Models" })).not.toBeInTheDocument();
  });

  it("renders the team detail overlay from the ?team drill-in", () => {
    detailState.teamId = "team-9";
    const { getByTestId } = renderPage();
    expect(getByTestId("team-info")).toHaveTextContent("team:team-9");
  });

  it("hides admin-only tabs for a non-admin user", () => {
    mockUseAuthorized.mockReturnValue(NON_ADMIN);
    const { queryByRole } = renderPage();
    expect(queryByRole("tab", { name: "LLM Credentials" })).not.toBeInTheDocument();
    expect(queryByRole("tab", { name: "Health Status" })).not.toBeInTheDocument();
  });

  // Auto-routers are excluded from the All Models table, so this tab is their home: the only
  // place in the product to list, create, edit or delete one.
  describe("Auto-Routers tab", () => {
    it("sits third, after All Models and Add Model", () => {
      const { getAllByRole } = renderPage();

      const tabs = getAllByRole("tab").map((tab) => tab.textContent);
      expect(tabs[0]).toContain("All Models");
      expect(tabs[1]).toBe("Add Model");
      expect(tabs[2]).toContain("Auto-Routers");
      // Badged Beta while the tab settles; BetaBadge renders the label text.
      expect(tabs[2]).toContain("Beta");
    });

    it("renders its panel when selected", async () => {
      const user = userEvent.setup();
      const { getByRole, getByTestId } = renderPage();

      await user.click(getByRole("tab", { name: /Auto-Routers/ }));
      expect(getByTestId("panel-auto-routers")).toBeInTheDocument();
    });

    it("is hidden from non-admins, who cannot write models", () => {
      mockUseAuthorized.mockReturnValue(NON_ADMIN);
      const { queryByRole } = renderPage();

      expect(queryByRole("tab", { name: /Auto-Routers/ })).not.toBeInTheDocument();
    });
  });
});
