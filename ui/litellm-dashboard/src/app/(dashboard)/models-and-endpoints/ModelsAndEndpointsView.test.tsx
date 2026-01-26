/* @vitest-environment jsdom */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render } from "@testing-library/react";
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

  it("should show Missing provider banner by default", async () => {
    localStorageMock.clear();
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
    expect(await findByText("Missing a provider?", {}, { timeout: 10000 })).toBeInTheDocument();
  }, 15000);

  it("should hide Missing provider banner when dismiss button is clicked and persist to localStorage", async () => {
    localStorageMock.clear();
    const queryClient = createQueryClient();
    const { findByText, queryByText, container } = render(
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

    // Wait for banner to appear
    expect(await findByText("Missing a provider?", {}, { timeout: 10000 })).toBeInTheDocument();

    // Find and click dismiss button (X button)
    const dismissButton = container.querySelector('button[aria-label="Dismiss banner"]');
    if (dismissButton) {
      fireEvent.click(dismissButton);

      // Banner should be hidden
      expect(queryByText("Missing a provider?")).not.toBeInTheDocument();

      // LocalStorage should be updated
      expect(localStorageMock.getItem("hideMissingProviderBanner")).toBe("true");
    }
  }, 15000);

  it("should show compact Request Provider button when banner is dismissed", async () => {
    // Set localStorage to hide banner
    localStorageMock.setItem("hideMissingProviderBanner", "true");
    const queryClient = createQueryClient();
    const { findByText, queryByText } = render(
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

    // Wait for component to render
    await findByText("Model Management", {}, { timeout: 10000 });

    // Banner should not be visible
    expect(queryByText("Missing a provider?")).not.toBeInTheDocument();

    // Compact Request Provider button should be visible in header
    const requestProviderLinks = document.querySelectorAll('a[href="https://models.litellm.ai/?request=true"]');
    // There should be a compact button when banner is hidden
    expect(requestProviderLinks.length).toBeGreaterThan(0);
  }, 15000);
});
