/* @vitest-environment jsdom */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, render } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import ModelsAndEndpointsView from "./ModelsAndEndpointsView";

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
        <ModelsAndEndpointsView
          token="123"
          modelData={{ data: [] }}
          keys={[]}
          setModelData={() => {}}
          premiumUser={false}
          teams={[]}
        />
      </QueryClientProvider>,
    );
    expect(await findByText("Model Management", {}, { timeout: 10000 })).toBeInTheDocument();
  }, 15000);

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
        <ModelsAndEndpointsView
          token="123"
          modelData={{ data: modelDataWithIds.data }}
          keys={[]}
          setModelData={() => {}}
          premiumUser={false}
          teams={[]}
        />
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
});
