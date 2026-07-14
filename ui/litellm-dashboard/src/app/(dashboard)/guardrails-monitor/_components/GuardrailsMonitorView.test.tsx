import { render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { beforeEach, describe, expect, it, vi } from "vitest";
import GuardrailsMonitorView from "./GuardrailsMonitorView";
import type { GuardrailUsageOverview } from "@/app/(dashboard)/hooks/guardrailsMonitor/useGuardrailsUsageOverview";

const useQueryMock = vi.fn();
vi.mock("@/lib/http/api", () => ({
  $api: { useQuery: (...args: unknown[]) => useQueryMock(...args) },
}));

const mockUseAuthorized = vi.fn();
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => mockUseAuthorized(),
}));

vi.mock("@/components/networking", () => ({
  formatDate: vi.fn((d: Date) => d.toISOString().slice(0, 10)),
  fetchAvailableModels: vi.fn(async () => []),
  uiSpendLogsCall: vi.fn(async () => ({ data: [], total: 0 })),
}));

const overview: GuardrailUsageOverview = {
  rows: [
    {
      id: "guardrail-1",
      name: "PII Detector",
      type: "Guardrail",
      provider: "Custom",
      requestsEvaluated: 120,
      failRate: 20,
      avgScore: null,
      avgLatency: null,
      status: "critical",
      trend: "up",
    },
  ],
  chart: [{ date: "2026-07-01", passed: 96, blocked: 24 }],
  totalRequests: 120,
  totalBlocked: 24,
  passRate: 80,
};

function wrapper({ children }: { children: React.ReactNode }) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
}

describe("GuardrailsMonitorView", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseAuthorized.mockReturnValue({ accessToken: "test-token", userRole: "Admin" });
    useQueryMock.mockReturnValue({ data: overview, isLoading: false, error: null });
  });

  it("requests the guardrails usage overview for the selected date range", async () => {
    render(<GuardrailsMonitorView accessToken="test-token" />, { wrapper });

    expect(await screen.findByRole("heading", { name: /Guardrails Monitor/i })).toBeDefined();
    await waitFor(() => {
      expect(useQueryMock).toHaveBeenCalledWith(
        "get",
        "/guardrails/usage/overview",
        expect.objectContaining({
          params: { query: { start_date: expect.any(String), end_date: expect.any(String) } },
        }),
        expect.objectContaining({ enabled: true }),
      );
    });
  });

  it("renders the typed overview rows and totals", async () => {
    render(<GuardrailsMonitorView accessToken="test-token" />, { wrapper });

    expect(await screen.findByText("PII Detector")).toBeDefined();
    expect(screen.getByText("20%")).toBeDefined();
    expect(screen.getByText("80%")).toBeDefined();
    expect(screen.getByText("critical")).toBeDefined();
  });

  it("disables the query when there is no access token", async () => {
    mockUseAuthorized.mockReturnValue({ accessToken: null, userRole: "Admin" });
    useQueryMock.mockReturnValue({ data: undefined, isLoading: false, error: null });

    render(<GuardrailsMonitorView accessToken={null} />, { wrapper });

    expect(await screen.findByRole("heading", { name: /Guardrails Monitor/i })).toBeDefined();
    expect(useQueryMock).toHaveBeenCalledWith(
      "get",
      "/guardrails/usage/overview",
      expect.anything(),
      expect.objectContaining({ enabled: false }),
    );
  });
});
