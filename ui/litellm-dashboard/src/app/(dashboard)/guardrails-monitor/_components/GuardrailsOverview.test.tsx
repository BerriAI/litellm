import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type {
  GuardrailUsageOverview,
  GuardrailUsageOverviewRow,
} from "@/app/(dashboard)/hooks/guardrails/useGuardrailsUsage";
import { GuardrailsOverview } from "./GuardrailsOverview";

const useGuardrailsUsageOverviewMock = vi.fn();
vi.mock("@/app/(dashboard)/hooks/guardrails/useGuardrailsUsage", () => ({
  useGuardrailsUsageOverview: (...args: unknown[]) => useGuardrailsUsageOverviewMock(...args),
}));

vi.mock("./ScoreChart", () => ({
  ScoreChart: () => <div>Score chart</div>,
}));

vi.mock("./EvaluationSettingsModal", () => ({
  EvaluationSettingsModal: ({ open }: { open: boolean }) => (open ? <div>Evaluation settings modal</div> : null),
}));

const baseRow: GuardrailUsageOverviewRow = {
  id: "guardrail",
  name: "Guardrail",
  type: "content_filter",
  provider: "LiteLLM",
  requestsEvaluated: 0,
  failRate: 0,
  avgScore: null,
  avgLatency: null,
  status: "healthy",
  trend: "stable",
  usageUnits: {},
  cost: null,
  untrackedUsageUnits: {},
};

const overview: GuardrailUsageOverview = {
  rows: [
    {
      ...baseRow,
      id: "guardrail-low",
      name: "Low Failure Guardrail",
      requestsEvaluated: 1200,
      failRate: 2.5,
      avgLatency: 45,
      trend: "down",
    },
    {
      ...baseRow,
      id: "guardrail-high",
      name: "High Failure Guardrail",
      provider: "Bedrock",
      requestsEvaluated: 300,
      failRate: 18,
      status: "warning",
      trend: "up",
      usageUnits: { contentPolicyUnits: 1000, sensitiveInformationPolicyUnits: 250 },
      cost: 0.15,
      untrackedUsageUnits: { sensitiveInformationPolicyUnits: 250 },
    },
    {
      ...baseRow,
      id: "guardrail-free",
      name: "Free Bedrock Guardrail",
      provider: "Bedrock",
      requestsEvaluated: 10,
      failRate: 0,
      usageUnits: { contentPolicyUnits: 40 },
      cost: 0,
    },
  ],
  chart: [],
  totalRequests: 1510,
  totalBlocked: 84,
  passRate: 94.4,
  totalUsageUnits: { contentPolicyUnits: 1040, sensitiveInformationPolicyUnits: 250 },
  totalCost: 0.15,
  totalUntrackedUsageUnits: { sensitiveInformationPolicyUnits: 250 },
};

function renderOverview(onSelectGuardrail = vi.fn()) {
  return render(
    <GuardrailsOverview
      accessToken="test-token"
      startDate="2026-08-01"
      endDate="2026-08-12"
      onSelectGuardrail={onSelectGuardrail}
    />,
  );
}

const rowNamed = (name: string) => screen.getByRole("row", { name: new RegExp(name) });

describe("GuardrailsOverview", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useGuardrailsUsageOverviewMock.mockReturnValue({ data: overview, isLoading: false, error: null });
  });

  it("asks for the usage overview of the selected window", () => {
    renderOverview();

    expect(useGuardrailsUsageOverviewMock).toHaveBeenCalledWith({
      accessToken: "test-token",
      startDate: "2026-08-01",
      endDate: "2026-08-12",
    });
  });

  it("renders performance data and selects a guardrail", async () => {
    const onSelectGuardrail = vi.fn();
    const user = userEvent.setup();

    renderOverview(onSelectGuardrail);

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

  it("shows each guardrail's usage units and cost, marking the units cost leaves out", async () => {
    renderOverview();

    expect(await screen.findByRole("columnheader", { name: "Usage Units" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: /Cost/ })).toBeInTheDocument();

    const priced = rowNamed("High Failure Guardrail");
    expect(within(priced).getByText("1,250")).toBeInTheDocument();
    expect(within(priced).getByText("$0.1500")).toBeInTheDocument();
    expect(within(priced).getByLabelText("250 units unpriced")).toBeInTheDocument();

    const free = rowNamed("Free Bedrock Guardrail");
    expect(within(free).getByText("40")).toBeInTheDocument();
    expect(within(free).getByText("$0.0000")).toBeInTheDocument();
    expect(within(free).queryByLabelText(/unpriced/)).not.toBeInTheDocument();

    const unmetered = rowNamed("Low Failure Guardrail");
    expect(within(unmetered).getAllByText("—")).toHaveLength(2);
  });

  it("breaks the usage units down per counter on hover", async () => {
    const user = userEvent.setup();
    renderOverview();

    await user.hover(within(rowNamed("High Failure Guardrail")).getByText("1,250"));

    expect(await screen.findByText("Content Policy: 1,000")).toBeInTheDocument();
    expect(screen.getByText("Sensitive Information Policy: 250")).toBeInTheDocument();
  });

  it("sorts by cost when its header is clicked, keeping guardrails with no known cost last either way", async () => {
    const user = userEvent.setup();
    renderOverview();
    const rowNames = () =>
      screen
        .getAllByRole("row")
        .slice(1)
        .map((r) => r.textContent ?? "");

    await user.click(await screen.findByRole("button", { name: /Cost/ }));
    await waitFor(() => expect(rowNames()[0]).toContain("Free Bedrock Guardrail"));
    expect(rowNames()[1]).toContain("High Failure Guardrail");
    expect(rowNames()[2]).toContain("Low Failure Guardrail");

    await user.click(screen.getByRole("button", { name: /Cost/ }));
    await waitFor(() => expect(rowNames()[0]).toContain("High Failure Guardrail"));
    expect(rowNames()[1]).toContain("Free Bedrock Guardrail");
    expect(rowNames()[2]).toContain("Low Failure Guardrail");
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

    expect(await screen.findByText("1,510")).toBeInTheDocument();
    expect(screen.getByText("Total Evaluations")).toBeInTheDocument();
    expect(screen.getByText("Blocked Requests")).toBeInTheDocument();
    expect(screen.getByText("84")).toBeInTheDocument();
    expect(screen.getByText("Pass Rate")).toBeInTheDocument();
    expect(screen.getByText("94.4%")).toBeInTheDocument();
    expect(screen.getByText("15ms")).toBeInTheDocument();
    expect(screen.getByText("Active Guardrails")).toBeInTheDocument();
    expect(screen.getByText("3")).toBeInTheDocument();
  });

  it("totals guardrail cost across the window and says how many units it leaves out", async () => {
    renderOverview();

    const card = await screen.findByRole("group", { name: "Guardrail Cost" });
    expect(card).toHaveTextContent("$0.1500");
    expect(card).toHaveTextContent("250 units unpriced");
  });

  it("lays the guardrail cost total out per guardrail", async () => {
    const user = userEvent.setup();
    renderOverview();

    const card = await screen.findByRole("group", { name: "Guardrail Cost" });
    await user.click(within(card).getByRole("button", { name: /How is this calculated/ }));

    const dialog = await screen.findByRole("dialog", { name: "How this cost is calculated" });
    const cells = within(dialog)
      .getAllByRole("row")
      .map((row) =>
        within(row)
          .getAllByRole("cell")
          .map((cell) => cell.textContent ?? ""),
      );
    expect(cells).toEqual([
      ["High Failure Guardrail", "$0.1500"],
      ["Free Bedrock Guardrail", "$0.0000"],
      ["Total", "$0.1500"],
    ]);
    expect(within(dialog).getByText(/250 units with no known price are left out of the cost/)).toBeInTheDocument();
    const issueLink = within(dialog).getByRole("link", { name: "Request pricing on GitHub" });
    const issueUrl = new URL(issueLink.getAttribute("href") ?? "");
    expect(issueUrl.searchParams.get("template")).toBe("feature_request.yml");
    expect(issueUrl.searchParams.get("the-feature")).toContain("sensitiveInformationPolicyUnits");
  });

  it("shows a dash for guardrail cost when nothing in the window was priced", async () => {
    useGuardrailsUsageOverviewMock.mockReturnValue({
      data: { ...overview, totalCost: null, totalUntrackedUsageUnits: {} },
      isLoading: false,
      error: null,
    });
    renderOverview();

    const card = await screen.findByRole("group", { name: "Guardrail Cost" });
    expect(card).toHaveTextContent("—");
    expect(card).not.toHaveTextContent("unpriced");
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
    useGuardrailsUsageOverviewMock.mockReturnValue({ data: undefined, isLoading: true, error: null });
    renderOverview();

    await waitFor(() => expect(document.querySelector('[aria-busy="true"]')).toBeInTheDocument());
  });

  it("shows a failure message when the usage request rejects", async () => {
    useGuardrailsUsageOverviewMock.mockReturnValue({
      data: undefined,
      isLoading: false,
      error: new Error("network down"),
    });
    renderOverview();

    expect(await screen.findByText("Failed to load data. Try again.")).toBeInTheDocument();
  });
});
