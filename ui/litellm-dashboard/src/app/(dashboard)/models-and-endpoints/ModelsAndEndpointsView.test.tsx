/* @vitest-environment jsdom */
import { render } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import ModelsAndEndpointsView from "./ModelsAndEndpointsView";

// Minimal stubs to avoid Next.js router and network usage during render
vi.mock("@/components/networking", () => ({
  credentialListCall: vi.fn().mockResolvedValue({ credentials: [] }),
  modelInfoCall: vi.fn().mockResolvedValue({ data: [] }),
  modelCostMap: vi.fn().mockResolvedValue({}),
  modelMetricsCall: vi.fn().mockResolvedValue({ data: [], all_api_bases: [] }),
  streamingModelMetricsCall: vi.fn().mockResolvedValue({ data: [], all_api_bases: [] }),
  modelExceptionsCall: vi.fn().mockResolvedValue({ data: [], exception_types: [] }),
  modelMetricsSlowResponsesCall: vi.fn().mockResolvedValue([]),
  getCallbacksCall: vi.fn().mockResolvedValue({ router_settings: {} }),
  setCallbacksCall: vi.fn().mockResolvedValue(undefined),
  modelSettingsCall: vi.fn().mockResolvedValue([]),
  adminGlobalActivityExceptions: vi.fn().mockResolvedValue({ sum_num_rate_limit_exceptions: 0, daily_data: [] }),
  adminGlobalActivityExceptionsPerDeployment: vi.fn().mockResolvedValue([]),
  allEndUsersCall: vi.fn().mockResolvedValue([]),
  latestHealthChecksCall: vi.fn().mockResolvedValue({ latest_health_checks: {} }),
  getPassThroughEndpointsCall: vi.fn().mockResolvedValue({ endpoints: {} }),
  getGuardrailsList: vi.fn().mockResolvedValue([]),
  tagListCall: vi.fn().mockResolvedValue([]),
  modelAvailableCall: vi.fn().mockResolvedValue({ data: [] }),
  modelHubCall: vi.fn().mockResolvedValue({ data: [] }),
  getModelCostMapReloadStatus: vi.fn().mockResolvedValue({
    scheduled: false,
    interval_hours: null,
    last_run: null,
    next_run: null,
  }),
}));

vi.mock("@/app/(dashboard)/models-and-endpoints/components/ModelAnalyticsTab/ModelAnalyticsTab", () => ({
  default: () => null,
}));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({
    token: "123",
    accessToken: "123",
    userId: "user-1",
    userEmail: "user@example.com",
    userRole: "Admin",
    premiumUser: false,
    disabledPersonalKeyCreation: null,
    showSSOBanner: false,
  }),
}));

vi.mock("@/app/(dashboard)/hooks/useTeams", () => ({
  default: () => ({
    teams: [],
    setTeams: vi.fn(),
  }),
}));

const createQueryClient = () =>
  new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } },
  });

describe("ModelsAndEndpointsView", () => {
  it("should render the models and endpoints view", async () => {
    // JSDOM polyfill for libraries expecting ResizeObserver (e.g., recharts)
    // Note: ResizeObserver is now globally mocked in setupTests.ts, but keeping this for backwards compatibility
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (global as any).ResizeObserver = class {
      observe() {}
      unobserve() {}
      disconnect() {}
    };
    const queryClient = createQueryClient();
    const { findByText } = render(
      <QueryClientProvider client={queryClient}>
        <ModelsAndEndpointsView
          accessToken="123"
          token="123"
          userRole="123"
          userID="123"
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
});
