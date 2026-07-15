/* @vitest-environment jsdom */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import ModelsAndEndpointsView from "./ModelsAndEndpointsView";

// Mock localStorage
const localStorageMock = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: (key: string) => store[key] || null,
    setItem: (key: string, value: string) => {
      store[key] = value;
    },
    removeItem: (key: string) => {
      delete store[key];
    },
    clear: () => {
      store = {};
    },
  };
})();
Object.defineProperty(window, "localStorage", { value: localStorageMock });

// Minimal stubs to avoid Next.js router and network usage during render
vi.mock("@/components/networking", () => ({
  credentialListCall: vi.fn().mockResolvedValue({ credentials: [] }),
  modelInfoCall: vi.fn().mockResolvedValue({ data: [] }),
  modelCostMap: vi.fn().mockResolvedValue({}),
  getPassThroughEndpointsCall: vi.fn().mockResolvedValue({ endpoints: {} }),
  getCallbacksCall: vi.fn().mockResolvedValue({ router_settings: {} }),
  setCallbacksCall: vi.fn().mockResolvedValue(undefined),
  getUiSettings: vi.fn().mockResolvedValue({ values: {} }),
  latestHealthChecksCall: vi.fn().mockResolvedValue({ latest_health_checks: {} }),
  getModelCostMapReloadStatus: vi.fn().mockResolvedValue({}),
}));

vi.mock("@/app/(dashboard)/models-and-endpoints/components/ModelAnalyticsTab/ModelAnalyticsTab", () => ({
  default: () => null,
}));

vi.mock("@/components/add_model/add_auto_router_tab", () => ({
  default: () => null,
}));

vi.mock("@/components/add_model/AddModelForm", () => ({
  default: () => null,
}));

const mockHealthCheckComponent = vi.fn((_props: { all_models_on_proxy?: string[] }) => null);
vi.mock("@/components/model_dashboard/HealthCheckComponent", () => ({
  default: (props: { all_models_on_proxy?: string[] }) => {
    mockHealthCheckComponent(props);
    return null;
  },
}));

vi.mock("@/app/(dashboard)/hooks/useTeams", () => ({
  default: () => ({
    teams: [],
    setTeams: vi.fn(),
  }),
}));

const mockUseModelsInfo = vi.fn();
vi.mock("@/app/(dashboard)/hooks/models/useModels", () => ({
  useModelsInfo: () => mockUseModelsInfo(),
}));

const mockUseUISettings = vi.fn();
vi.mock("@/app/(dashboard)/hooks/uiSettings/useUISettings", () => ({
  useUISettings: () => mockUseUISettings(),
}));

const mockUseModelCostMap = vi.fn();
vi.mock("@/app/(dashboard)/hooks/models/useModelCostMap", () => ({
  useModelCostMap: () => mockUseModelCostMap(),
}));

const mockUseAuthorized = vi.fn();
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => mockUseAuthorized(),
}));

const createQueryClient = () =>
  new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } },
  });

describe("ModelsAndEndpointsView", () => {
  beforeEach(() => {
    mockUseModelsInfo.mockReturnValue({
      data: { data: [] },
      isLoading: false,
      refetch: vi.fn(),
    });
    mockUseUISettings.mockReturnValue({
      data: { values: {} },
    });
    mockUseModelCostMap.mockReturnValue({
      data: {},
      isLoading: false,
      error: null,
    });
    mockUseAuthorized.mockReturnValue({
      accessToken: "123",
      token: "123",
      userRole: "Admin",
      userId: "123",
    });
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (global as any).ResizeObserver = class {
      observe() {}
      unobserve() {}
      disconnect() {}
    };
  });

  it("should render the models and endpoints view", async () => {
    const queryClient = createQueryClient();
    const { findByText } = render(
      <QueryClientProvider client={queryClient}>
        <ModelsAndEndpointsView premiumUser={false} teams={[]} />
      </QueryClientProvider>,
    );
    expect(await findByText("Model Management", {}, { timeout: 10000 })).toBeInTheDocument();
  });

  it("should show Cost Optimization feedback banner by default", async () => {
    localStorageMock.clear();
    const queryClient = createQueryClient();
    const { findByText } = render(
      <QueryClientProvider client={queryClient}>
        <ModelsAndEndpointsView premiumUser={false} teams={[]} />
      </QueryClientProvider>,
    );
    expect(await findByText("Help shape cost optimization", {}, { timeout: 10000 })).toBeInTheDocument();
  });

  it("should hide Cost Optimization feedback banner when dismiss button is clicked and persist to localStorage", async () => {
    localStorageMock.clear();
    const queryClient = createQueryClient();
    const { findByText, queryByText, container } = render(
      <QueryClientProvider client={queryClient}>
        <ModelsAndEndpointsView premiumUser={false} teams={[]} />
      </QueryClientProvider>,
    );

    // Wait for banner to appear
    expect(await findByText("Help shape cost optimization", {}, { timeout: 10000 })).toBeInTheDocument();

    // Find and click dismiss button (X button)
    const dismissButton = container.querySelector('button[aria-label="Dismiss banner"]');
    expect(dismissButton).not.toBeNull();
    fireEvent.click(dismissButton!);

    // Banner should be hidden
    expect(queryByText("Help shape cost optimization")).not.toBeInTheDocument();

    // LocalStorage should be updated
    expect(localStorageMock.getItem("hideCostOptimizationFeedbackBanner")).toBe("true");
  });

  it("should keep Cost Optimization feedback banner hidden across remounts once dismissed", async () => {
    // Set localStorage to hide banner
    localStorageMock.setItem("hideCostOptimizationFeedbackBanner", "true");
    const queryClient = createQueryClient();
    const { findByText, queryByText } = render(
      <QueryClientProvider client={queryClient}>
        <ModelsAndEndpointsView premiumUser={false} teams={[]} />
      </QueryClientProvider>,
    );

    // Wait for component to render
    await findByText("Model Management", {}, { timeout: 10000 });

    // Banner should not be visible
    expect(queryByText("Help shape cost optimization")).not.toBeInTheDocument();
  });

  it("should pass model IDs (not model names) to HealthCheckComponent as all_models_on_proxy", async () => {
    mockHealthCheckComponent.mockClear();
    const modelDataWithIds = {
      data: [
        { model_name: "gpt-4", model_info: { id: "deployment-id-1" } },
        { model_name: "gpt-4", model_info: { id: "deployment-id-2" } },
      ],
    };
    mockUseModelsInfo.mockReturnValue({
      data: { data: modelDataWithIds.data },
      isLoading: false,
      refetch: vi.fn(),
    });

    const queryClient = createQueryClient();
    const { getByRole } = render(
      <QueryClientProvider client={queryClient}>
        <ModelsAndEndpointsView premiumUser={false} teams={[]} />
      </QueryClientProvider>,
    );

    const healthStatusTab = getByRole("tab", { name: "Health Status" });
    await act(async () => {
      healthStatusTab.click();
    });

    expect(mockHealthCheckComponent).toHaveBeenCalled();
    const healthCheckProps = mockHealthCheckComponent.mock.calls[0][0];
    expect(healthCheckProps.all_models_on_proxy).toEqual(["deployment-id-1", "deployment-id-2"]);
    expect(healthCheckProps.all_models_on_proxy).not.toContain("gpt-4");
  });

  it("should hide Add Model from an Internal Viewer who is a team admin", async () => {
    const internalViewerAuth = {
      accessToken: "123",
      token: "123",
      userRole: "Internal Viewer",
      userId: "viewer-123",
    };
    mockUseAuthorized.mockReturnValue(internalViewerAuth);
    const teams = [
      {
        team_id: "team-1",
        team_alias: "Team 1",
        models: [],
        max_budget: null,
        budget_duration: null,
        tpm_limit: null,
        rpm_limit: null,
        organization_id: null,
        created_at: "2024-01-01",
        keys: [],
        members_with_roles: [{ user_id: "viewer-123", role: "admin" }],
      },
    ];

    const queryClient = createQueryClient();
    const { findByText } = render(
      <QueryClientProvider client={queryClient}>
        <ModelsAndEndpointsView premiumUser teams={teams} />
      </QueryClientProvider>,
    );

    expect(await findByText("Your Models", {}, { timeout: 10000 })).toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: "Add Model" })).not.toBeInTheDocument();
  });
});
