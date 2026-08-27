import { render, screen, waitFor } from "@testing-library/react";
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
  EvaluationSettingsModal: ({ open }: { open: boolean }) => (open ? <div>Evaluation settings modal</div> : null),
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

function renderOverview(onSelectGuardrail = vi.fn()) {
  return render(
    <GuardrailsOverview
      accessToken="test-token"
      startDate="2026-08-01"
      endDate="2026-08-12"
      onSelectGuardrail={onSelectGuardrail}
    />,
    { wrapper },
  );
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

  it("renders the page header and the export action", async () => {
    renderOverview();

    expect(await screen.findByRole("heading", { name: "Guardrails Monitor", level: 1 })).toBeInTheDocument();
    expect(screen.getByText("Monitor guardrail performance across all requests")).toBeInTheDocument();
    expect(document.querySelector(".lucide-heart-pulse")).not.toBeNull();
    expect(screen.getByRole("button", { name: /Export Data/i })).toBeInTheDocument();
  });

  it("renders every summary metric card", async () => {
    renderOverview();

    expect(await screen.findByText("1,500")).toBeInTheDocument();
    expect(screen.getByText("Total Evaluations")).toBeInTheDocument();
    expect(screen.getByText("Blocked Requests")).toBeInTheDocument();
    expect(screen.getByText("84")).toBeInTheDocument();
    expect(screen.getByText("Pass Rate")).toBeInTheDocument();
    expect(screen.getByText("94.4%")).toBeInTheDocument();
    expect(screen.getByText("23ms")).toBeInTheDocument();
    expect(screen.getByText("Active Guardrails")).toBeInTheDocument();
    expect(screen.getByText("2")).toBeInTheDocument();
  });

  it("renders the table toolbar heading and its description", async () => {
    renderOverview();

    expect(await screen.findByRole("heading", { name: "Guardrail Performance", level: 5 })).toBeInTheDocument();
    expect(screen.getByText("Click a guardrail to view details, logs, and configuration")).toBeInTheDocument();
  });

  it("opens the evaluation settings modal from the toolbar action", async () => {
    const user = userEvent.setup();
    renderOverview();

    expect(screen.queryByText("Evaluation settings modal")).not.toBeInTheDocument();

    await user.click(await screen.findByTitle("Evaluation settings"));

    expect(await screen.findByText("Evaluation settings modal")).toBeInTheDocument();
  });

  it("marks the overview busy while the usage request is in flight", async () => {
    mockGetGuardrailsUsageOverview.mockReturnValue(new Promise(() => {}));
    renderOverview();

    await waitFor(() => expect(document.querySelector('[aria-busy="true"]')).toBeInTheDocument());
  });

  it("shows a failure message when the usage request rejects", async () => {
    mockGetGuardrailsUsageOverview.mockRejectedValue(new Error("network down"));
    renderOverview();

    expect(await screen.findByText("Failed to load data. Try again.")).toBeInTheDocument();
  });
});
