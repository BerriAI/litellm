import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import * as networking from "@/components/networking";
import { GuardrailsOverview } from "./GuardrailsOverview";

vi.mock("@/components/networking", () => ({
  getGuardrailsUsageOverview: vi.fn(),
}));

vi.mock("./ScoreChart", () => ({
  ScoreChart: () => <div>Score chart</div>,
}));

vi.mock("./EvaluationSettingsModal", () => ({
  EvaluationSettingsModal: () => null,
}));

const mockGetGuardrailsUsageOverview = vi.mocked(networking.getGuardrailsUsageOverview);

function wrapper({ children }: { children: React.ReactNode }) {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  });
  return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
}

describe("GuardrailsOverview", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockGetGuardrailsUsageOverview.mockResolvedValue({
      rows: [
        {
          id: "guardrail-low",
          name: "Low Failure Guardrail",
          type: "content_filter",
          provider: "LiteLLM",
          requestsEvaluated: 1200,
          failRate: 2.5,
          avgLatency: 45,
          status: "healthy",
          trend: "down",
        },
        {
          id: "guardrail-high",
          name: "High Failure Guardrail",
          type: "content_filter",
          provider: "Bedrock",
          requestsEvaluated: 300,
          failRate: 18,
          status: "warning",
          trend: "up",
        },
      ],
      chart: [],
      totalRequests: 1500,
      totalBlocked: 84,
      passRate: 94.4,
    });
  });

  it("renders performance data and selects a guardrail", async () => {
    const onSelectGuardrail = vi.fn();
    const user = userEvent.setup();

    render(
      <GuardrailsOverview
        accessToken="test-token"
        startDate="2026-08-01"
        endDate="2026-08-12"
        onSelectGuardrail={onSelectGuardrail}
      />,
      { wrapper },
    );

    expect(await screen.findByRole("columnheader", { name: "Guardrail" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: /Requests/ })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: /Fail Rate/ })).toBeInTheDocument();
    expect(await screen.findByText("Low Failure Guardrail")).toBeInTheDocument();
    expect(screen.getByText("1,200")).toBeInTheDocument();
    expect(screen.getByText("18%")).toBeInTheDocument();
    expect(screen.getByText("45ms")).toBeInTheDocument();

    const rows = screen.getAllByRole("row");
    expect(rows[1]).toHaveTextContent("High Failure Guardrail");
    expect(rows[2]).toHaveTextContent("Low Failure Guardrail");

    await user.click(screen.getByRole("button", { name: "Low Failure Guardrail" }));

    expect(onSelectGuardrail).toHaveBeenCalledWith("guardrail-low");
  });
});
