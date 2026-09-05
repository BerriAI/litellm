import { render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, expect, it, vi } from "vitest";
import GuardrailsMonitorView from "./GuardrailsMonitorView";

vi.mock("@/components/networking", () => ({
  formatDate: vi.fn((d: Date) => d.toISOString().slice(0, 10)),
}));

const mockUseGuardrailsUsageOverview = vi.fn();
vi.mock("@/app/(dashboard)/hooks/guardrails/useGuardrailsUsage", () => ({
  useGuardrailsUsageOverview: (...args: unknown[]) => mockUseGuardrailsUsageOverview(...args),
}));

function wrapper({ children }: { children: React.ReactNode }) {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  });
  return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
}

describe("GuardrailsMonitorView", () => {
  it("should render overview and fetch guardrails usage when accessToken is provided", async () => {
    mockUseGuardrailsUsageOverview.mockReturnValue({ data: undefined, isLoading: true, error: null });

    render(<GuardrailsMonitorView accessToken="test-token" />, { wrapper });

    expect(await screen.findByRole("heading", { name: /Guardrails Monitor/i })).toBeInTheDocument();
    await waitFor(() => {
      expect(mockUseGuardrailsUsageOverview).toHaveBeenCalledWith(
        expect.objectContaining({ accessToken: "test-token", startDate: expect.any(String) }),
      );
    });
  });

  it("should render without crashing when accessToken is null", async () => {
    mockUseGuardrailsUsageOverview.mockReturnValue({ data: undefined, isLoading: false, error: null });
    render(<GuardrailsMonitorView accessToken={null} />, { wrapper });
    expect(await screen.findByRole("heading", { name: /Guardrails Monitor/i })).toBeInTheDocument();
  });
});
