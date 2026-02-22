import { render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, expect, it, vi } from "vitest";
import GuardrailsMonitorView from "./GuardrailsMonitorView";
import * as networking from "@/components/networking";

vi.mock("@/components/networking", () => ({
  getGuardrailsUsageOverview: vi.fn(),
  getPoliciesUsageOverview: vi.fn(),
  formatDate: vi.fn((d: Date) => d.toISOString().slice(0, 10)),
}));

const mockGetGuardrailsUsageOverview = vi.mocked(networking.getGuardrailsUsageOverview);
const mockGetPoliciesUsageOverview = vi.mocked(networking.getPoliciesUsageOverview);

function wrapper({ children }: { children: React.ReactNode }) {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  });
  return (
    <QueryClientProvider client={queryClient}>
      {children}
    </QueryClientProvider>
  );
}

describe("GuardrailsMonitorView", () => {
  it("should render overview and fetch guardrails and policies usage when accessToken is provided", async () => {
    mockGetGuardrailsUsageOverview.mockResolvedValue({
      rows: [],
      chart: [],
      totalRequests: 0,
      totalBlocked: 0,
      passRate: 100,
    });
    mockGetPoliciesUsageOverview.mockResolvedValue({
      rows: [],
      chart: [],
      totalRequests: 0,
      totalBlocked: 0,
      passRate: 100,
    });

    render(
      <GuardrailsMonitorView accessToken="test-token" />,
      { wrapper }
    );

    expect(await screen.findByRole("heading", { name: /Guardrails Monitor/i })).toBeDefined();
    await waitFor(() => {
      expect(mockGetGuardrailsUsageOverview).toHaveBeenCalled();
      expect(mockGetPoliciesUsageOverview).toHaveBeenCalled();
    });
  });

  it("should render without crashing when accessToken is null", async () => {
    render(<GuardrailsMonitorView accessToken={null} />, { wrapper });
    expect(await screen.findByRole("heading", { name: /Guardrails Monitor/i })).toBeDefined();
  });
});
