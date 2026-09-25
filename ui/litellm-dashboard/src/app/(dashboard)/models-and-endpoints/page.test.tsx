/* @vitest-environment jsdom */
import { render, renderWithProviders, screen, testQueryClient, waitFor } from "@/../tests/test-utils";
import type { Team } from "@/components/networking";
import { QueryClientProvider } from "@tanstack/react-query";
import userEvent from "@testing-library/user-event";
import { NuqsTestingAdapter, type OnUrlUpdateFunction } from "nuqs/adapters/testing";
import type { PropsWithChildren } from "react";
import { beforeEach, describe, expect, it, type Mock, vi } from "vitest";
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
const teamInfoProps = vi.hoisted(() => vi.fn());
vi.mock("@/components/team/TeamInfo", () => ({
  default: (props: { teamId: string; is_team_admin: boolean; is_proxy_admin: boolean }) => {
    teamInfoProps(props);
    return (
      <div data-testid="team-info" data-team-admin={String(props.is_team_admin)}>
        team:{props.teamId}
      </div>
    );
  },
}));

const mockUseAuthorized = vi.fn();
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({ default: () => mockUseAuthorized() }));
const teamsState: { data: Team[] | undefined; isLoading: boolean } = { data: [], isLoading: false };
vi.mock("@/app/(dashboard)/hooks/teams/useTeams", () => ({
  useTeams: () => ({ data: teamsState.data, isLoading: teamsState.isLoading }),
}));
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
// A proxy_admin_viewer session: effectiveSessionRole masquerades the role as "Admin".
const VIEW_ONLY_ADMIN = { ...ADMIN, isViewOnly: true };

const TEAM_ADMINED_BY_U1 = {
  team_id: "team-1",
  team_alias: "Engineering",
  members_with_roles: [{ user_id: "u1", role: "admin" }],
} as unknown as Team;

interface RenderPageOptions {
  observeMountUrlWrites?: boolean;
}

const renderPage = (searchParams?: string, { observeMountUrlWrites = false }: RenderPageOptions = {}) => {
  const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
  if (!observeMountUrlWrites) {
    const view = renderWithProviders(<ModelsAndEndpointsPage />, { searchParams, onUrlUpdate });
    return { ...view, onUrlUpdate };
  }
  const MountPreservingProviders = ({ children }: PropsWithChildren) => (
    <NuqsTestingAdapter
      searchParams={searchParams}
      onUrlUpdate={onUrlUpdate}
      hasMemory
      resetUrlUpdateQueueOnMount={false}
    >
      <QueryClientProvider client={testQueryClient}>{children}</QueryClientProvider>
    </NuqsTestingAdapter>
  );
  const view = render(<ModelsAndEndpointsPage />, { wrapper: MountPreservingProviders });
  return { ...view, onUrlUpdate };
};

const lastUrl = (onUrlUpdate: Mock<OnUrlUpdateFunction>): URLSearchParams => {
  const event = onUrlUpdate.mock.calls.at(-1)?.[0];
  if (!event) throw new Error("no URL update was emitted");
  return event.searchParams;
};

const URL_SETTLE_MS = 100;

describe("ModelsAndEndpointsPage", () => {
  beforeEach(() => {
    detailState.modelId = null;
    detailState.teamId = null;
    teamsState.data = [];
    teamsState.isLoading = false;
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

  it("switches tabs, mounting only the active panel", async () => {
    const user = userEvent.setup();
    renderPage();
    await user.click(screen.getByRole("tab", { name: "Health Status" }));
    expect(await screen.findByTestId("panel-health")).toBeInTheDocument();
    expect(screen.queryByTestId("panel-all-models")).not.toBeInTheDocument();
  });

  describe("?tab= deep links", () => {
    it("opens the tab named in the URL on load", () => {
      renderPage("?tab=health");

      expect(screen.getByRole("tab", { name: "Health Status" })).toHaveAttribute("aria-selected", "true");
      expect(screen.getByTestId("panel-health")).toBeInTheDocument();
      expect(screen.queryByTestId("panel-all-models")).not.toBeInTheDocument();
    });

    it("leaves a URL that names a visible tab untouched", async () => {
      const { onUrlUpdate } = renderPage("?tab=llm-credentials", { observeMountUrlWrites: true });

      await new Promise((resolve) => setTimeout(resolve, URL_SETTLE_MS));

      expect(onUrlUpdate).not.toHaveBeenCalled();
      expect(screen.getByTestId("panel-credentials")).toBeInTheDocument();
    });

    it("writes the selected tab to ?tab= and drops the key when All Models is reselected", async () => {
      const user = userEvent.setup();
      const { onUrlUpdate } = renderPage();

      await user.click(screen.getByRole("tab", { name: "Health Status" }));
      await waitFor(() => expect(lastUrl(onUrlUpdate).get("tab")).toBe("health"));

      await user.click(screen.getByRole("tab", { name: "All Models" }));
      await waitFor(() => expect(lastUrl(onUrlUpdate).has("tab")).toBe(false));
      expect(screen.getByTestId("panel-all-models")).toBeInTheDocument();
    });

    it("falls back to All Models and clears ?tab= for a tab the role cannot see", async () => {
      mockUseAuthorized.mockReturnValue(NON_ADMIN);
      const { onUrlUpdate } = renderPage("?tab=health&model_search=gpt", { observeMountUrlWrites: true });

      expect(screen.getByTestId("panel-all-models")).toBeInTheDocument();
      expect(screen.queryByTestId("panel-health")).not.toBeInTheDocument();
      await waitFor(() => expect(onUrlUpdate).toHaveBeenCalled());
      expect(lastUrl(onUrlUpdate).has("tab")).toBe(false);
      expect(lastUrl(onUrlUpdate).get("model_search")).toBe("gpt");
    });

    it("falls back to All Models and clears an unknown ?tab=", async () => {
      const { onUrlUpdate } = renderPage("?tab=settings", { observeMountUrlWrites: true });

      expect(screen.getByTestId("panel-all-models")).toBeInTheDocument();
      await waitFor(() => expect(onUrlUpdate).toHaveBeenCalled());
      expect(lastUrl(onUrlUpdate).has("tab")).toBe(false);
    });

    it("hides the write-only tabs from a view-only admin deep link", async () => {
      mockUseAuthorized.mockReturnValue(VIEW_ONLY_ADMIN);
      const { onUrlUpdate } = renderPage("?tab=llm-credentials", { observeMountUrlWrites: true });

      expect(screen.getByTestId("panel-all-models")).toBeInTheDocument();
      await waitFor(() => expect(lastUrl(onUrlUpdate).has("tab")).toBe(false));
    });

    it("holds a team-scoped ?tab= while teams load, then opens it once the user is a team admin", async () => {
      mockUseAuthorized.mockReturnValue(NON_ADMIN);
      teamsState.data = undefined;
      teamsState.isLoading = true;
      const { onUrlUpdate, rerender } = renderPage("?tab=add", { observeMountUrlWrites: true });

      await new Promise((resolve) => setTimeout(resolve, URL_SETTLE_MS));
      expect(onUrlUpdate).not.toHaveBeenCalled();
      expect(screen.queryByTestId("panel-all-models")).not.toBeInTheDocument();

      teamsState.data = [TEAM_ADMINED_BY_U1];
      teamsState.isLoading = false;
      rerender(<ModelsAndEndpointsPage />);

      expect(await screen.findByTestId("panel-add")).toBeInTheDocument();
      expect(screen.getByRole("tab", { name: "Add Model" })).toHaveAttribute("aria-selected", "true");
      expect(onUrlUpdate).not.toHaveBeenCalled();
    });

    it("clears a team-scoped ?tab= once teams load and the user cannot create models", async () => {
      mockUseAuthorized.mockReturnValue(NON_ADMIN);
      teamsState.data = undefined;
      teamsState.isLoading = true;
      const { onUrlUpdate, rerender } = renderPage("?tab=add");

      teamsState.data = [];
      teamsState.isLoading = false;
      rerender(<ModelsAndEndpointsPage />);

      expect(await screen.findByTestId("panel-all-models")).toBeInTheDocument();
      await waitFor(() => expect(onUrlUpdate).toHaveBeenCalled());
      expect(lastUrl(onUrlUpdate).has("tab")).toBe(false);
    });

    it.each(["add", "auto-routers"])(
      "falls back to Your Models and clears ?tab=%s when the teams request failed",
      async (tab) => {
        mockUseAuthorized.mockReturnValue(NON_ADMIN);
        teamsState.data = undefined;
        teamsState.isLoading = false;
        const { onUrlUpdate } = renderPage(`?tab=${tab}`, { observeMountUrlWrites: true });

        expect(screen.getByTestId("panel-all-models")).toBeInTheDocument();
        expect(screen.getByRole("tab", { name: "Your Models" })).toHaveAttribute("aria-selected", "true");
        await waitFor(() => expect(lastUrl(onUrlUpdate).has("tab")).toBe(false));
      },
    );

    it("does not hold ?tab=add for a view-only admin while teams load", async () => {
      mockUseAuthorized.mockReturnValue(VIEW_ONLY_ADMIN);
      teamsState.data = undefined;
      teamsState.isLoading = true;
      const { onUrlUpdate } = renderPage("?tab=add", { observeMountUrlWrites: true });

      expect(screen.getByTestId("panel-all-models")).toBeInTheDocument();
      expect(screen.getByRole("tab", { name: "All Models" })).toHaveAttribute("aria-selected", "true");
      await waitFor(() => expect(lastUrl(onUrlUpdate).has("tab")).toBe(false));
    });
  });

  it("renders the model detail overlay from the ?model drill-in and hides the tabs", () => {
    detailState.modelId = "abc-123";
    renderPage();
    expect(screen.getByTestId("model-info")).toHaveTextContent("model:abc-123");
    expect(screen.queryByRole("tab", { name: "All Models" })).not.toBeInTheDocument();
  });

  it("renders the team detail overlay from the ?team drill-in with admin edit rights", () => {
    detailState.teamId = "team-9";
    renderPage();
    expect(screen.getByTestId("team-info")).toHaveTextContent("team:team-9");
    expect(screen.getByTestId("team-info")).toHaveAttribute("data-team-admin", "true");
  });

  it("passes is_proxy_admin for an admin session on the ?team drill-in", () => {
    detailState.teamId = "team-a1b2";
    renderPage();
    expect(teamInfoProps).toHaveBeenLastCalledWith(
      expect.objectContaining({ is_proxy_admin: true, is_team_admin: true }),
    );
  });

  it("opens the ?team drill-in without edit rights for a view-only admin", () => {
    mockUseAuthorized.mockReturnValue(VIEW_ONLY_ADMIN);
    detailState.teamId = "team-9";
    renderPage();
    expect(screen.getByTestId("team-info")).toHaveTextContent("team:team-9");
    expect(screen.getByTestId("team-info")).toHaveAttribute("data-team-admin", "false");
    expect(teamInfoProps).toHaveBeenLastCalledWith(expect.objectContaining({ is_proxy_admin: false }));
  });

  it("hides admin-only tabs for a non-admin user", () => {
    mockUseAuthorized.mockReturnValue(NON_ADMIN);
    renderPage();
    expect(screen.queryByRole("tab", { name: "LLM Credentials" })).not.toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: "Health Status" })).not.toBeInTheDocument();
  });

  it("keeps the full admin tab order for a real admin", () => {
    renderPage();
    expect(screen.getAllByRole("tab").map((tab) => tab.textContent)).toEqual([
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
    renderPage();
    expect(screen.getByRole("tab", { name: "All Models" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Health Status" })).toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: "LLM Credentials" })).not.toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: "Pass-Through Endpoints" })).not.toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: "Model Retry Settings" })).not.toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: "Model Group Alias" })).not.toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: /Model Access Group Budgets/ })).not.toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: "Price Data Reload" })).not.toBeInTheDocument();
  });

  // POST /model/new 403s a proxy_admin_viewer, so the form's tab must not render for one.
  it("hides the Add Model tab for a view-only admin session", () => {
    mockUseAuthorized.mockReturnValue(VIEW_ONLY_ADMIN);
    renderPage();
    expect(screen.queryByRole("tab", { name: "Add Model" })).not.toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "All Models" })).toBeInTheDocument();
  });

  // Read parity: the Auto-Routers list stays reachable for a view-only admin; only the
  // create affordance inside it is withheld, which AutoRoutersTabPanel decides.
  it("keeps the Auto-Routers tab for a view-only admin session", () => {
    mockUseAuthorized.mockReturnValue(VIEW_ONLY_ADMIN);
    renderPage();
    expect(screen.getByRole("tab", { name: /Auto-Routers/ })).toBeInTheDocument();
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
      expect(await screen.findByTestId("panel-auto-routers")).toBeInTheDocument();
    });

    it("is hidden from non-admins, who cannot write models", () => {
      mockUseAuthorized.mockReturnValue(NON_ADMIN);
      renderPage();

      expect(screen.queryByRole("tab", { name: /Auto-Routers/ })).not.toBeInTheDocument();
    });
  });
});
