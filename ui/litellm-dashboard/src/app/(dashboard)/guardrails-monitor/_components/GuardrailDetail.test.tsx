import { describe, it, expect, vi, beforeEach } from "vitest";
import userEvent from "@testing-library/user-event";
import type { UrlUpdateEvent } from "nuqs/adapters/testing";
import { renderWithProviders, screen, testQueryClient, waitFor } from "@/../tests/test-utils";
import type { GuardrailUsageDetail } from "@/app/(dashboard)/hooks/guardrails/useGuardrailsUsage";
import type { LogViewerState } from "@/components/GuardrailsMonitor/useLogViewerState";
import { GuardrailDetail } from "./GuardrailDetail";

const mockUseGuardrailsUsageDetail = vi.fn();
vi.mock("@/app/(dashboard)/hooks/guardrails/useGuardrailsUsage", () => ({
  useGuardrailsUsageDetail: (...args: unknown[]) => mockUseGuardrailsUsageDetail(...args),
}));

const mockGetGuardrailsUsageLogs = vi.fn();
vi.mock("@/components/networking", () => ({
  getGuardrailsUsageLogs: (...args: unknown[]) => mockGetGuardrailsUsageLogs(...args),
}));

vi.mock("@/components/GuardrailsMonitor/LogViewer", () => ({
  LogViewer: ({ guardrailName, viewState }: { guardrailName: string; viewState?: LogViewerState }) => (
    <div
      data-testid="log-viewer"
      data-filter={viewState?.filter}
      data-sample={viewState?.sampleSize}
      data-request={viewState?.requestId ?? ""}
    >
      <span>{guardrailName}</span>
      <button onClick={() => viewState?.setFilter("blocked")}>show blocked</button>
      <button onClick={() => viewState?.setSampleSize(50)}>sample 50</button>
      <button onClick={() => viewState?.setRequestId("req-9")}>open req-9</button>
      <button onClick={() => viewState?.setRequestId(null)}>close request</button>
    </div>
  ),
}));

vi.mock("./EvaluationSettingsModal", () => ({
  EvaluationSettingsModal: ({ open }: { open: boolean }) => (open ? <div data-testid="evaluation-modal" /> : null),
}));

const detail: GuardrailUsageDetail = {
  guardrail_id: "pii-detector",
  guardrail_name: "pii-detector",
  description: "Blocks personally identifiable information",
  status: "warning",
  provider: "presidio",
  type: "pii",
  requestsEvaluated: 12345,
  failRate: 20,
  avgScore: 0.4,
  avgLatency: 180,
  trend: "stable",
  time_series: [],
  usage_units: { sensitiveInformationPolicyUnits: 4 },
  usage_units_daily: [],
  usage_units_by_team: { "": { sensitiveInformationPolicyUnits: 4 } },
  usage_units_by_key: { "hash-1": { sensitiveInformationPolicyUnits: 4 } },
  cost: 0.0004,
  cost_by_unit: { sensitiveInformationPolicyUnits: 0.0004 },
  cost_by_team: { "": 0.0004 },
  cost_by_key: { "hash-1": 0.0004 },
  untracked_usage_units: {},
  untracked_usage_units_by_team: {},
  untracked_usage_units_by_key: {},
};

const loaded = (data: GuardrailUsageDetail | undefined) => ({ data, isLoading: false, error: null });

const defaultProps = {
  guardrailId: "pii-detector",
  onBack: vi.fn(),
  accessToken: "test-token" as string | null,
  startDate: "2026-07-01",
  endDate: "2026-07-24",
};

function renderDetail(
  props: Partial<typeof defaultProps> = {},
  urlOptions: { searchParams?: string; onUrlUpdate?: (event: UrlUpdateEvent) => void } = {},
) {
  return renderWithProviders(<GuardrailDetail {...defaultProps} {...props} />, urlOptions);
}

const lastUrlUpdate = (onUrlUpdate: ReturnType<typeof vi.fn<(event: UrlUpdateEvent) => void>>) =>
  onUrlUpdate.mock.calls.at(-1)?.[0];

describe("GuardrailDetail", () => {
  beforeEach(() => {
    testQueryClient.clear();
    vi.clearAllMocks();
    mockUseGuardrailsUsageDetail.mockReturnValue(loaded(detail));
    mockGetGuardrailsUsageLogs.mockResolvedValue({ logs: [], total: 0 });
  });

  it("should show a busy indicator while the detail request is in flight", () => {
    mockUseGuardrailsUsageDetail.mockReturnValue({ data: undefined, isLoading: true, error: null });
    renderDetail();
    expect(document.querySelector('[aria-busy="true"]')).toBeInTheDocument();
    expect(screen.queryByText("pii-detector")).not.toBeInTheDocument();
  });

  it("should show an error message and a way back when the detail request fails", async () => {
    mockUseGuardrailsUsageDetail.mockReturnValue({ data: undefined, isLoading: false, error: new Error("boom") });
    renderDetail();
    expect(await screen.findByText("Failed to load guardrail details.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /back to overview/i })).toBeInTheDocument();
  });

  it("should request the detail and the logs for the guardrail and date range", async () => {
    renderDetail();
    expect(mockUseGuardrailsUsageDetail).toHaveBeenCalledWith("pii-detector", {
      accessToken: "test-token",
      startDate: "2026-07-01",
      endDate: "2026-07-24",
    });
    await waitFor(() => expect(mockGetGuardrailsUsageLogs).toHaveBeenCalled());
    expect(mockGetGuardrailsUsageLogs).toHaveBeenCalledWith(
      "test-token",
      expect.objectContaining({ guardrailId: "pii-detector", startDate: "2026-07-01", endDate: "2026-07-24" }),
    );
  });

  it("should show the guardrail name, description, provider and capitalised status", async () => {
    renderDetail();
    expect(await screen.findByRole("heading", { name: "pii-detector" })).toBeInTheDocument();
    expect(screen.getByText("Blocks personally identifiable information")).toBeInTheDocument();
    expect(screen.getByText("presidio")).toBeInTheDocument();
    expect(screen.getByText("Warning")).toBeInTheDocument();
  });

  it("should show the usage metrics with the blocked count derived from the fail rate", async () => {
    renderDetail();
    expect(await screen.findByText("12,345")).toBeInTheDocument();
    expect(screen.getByText("20%")).toBeInTheDocument();
    expect(screen.getByText("2,469 blocked")).toBeInTheDocument();
    expect(screen.getByText("180ms")).toBeInTheDocument();
  });

  it("should show a placeholder when no latency has been recorded", async () => {
    mockUseGuardrailsUsageDetail.mockReturnValue(loaded({ ...detail, avgLatency: null }));
    renderDetail();
    expect(await screen.findByText("No data")).toBeInTheDocument();
  });

  it("should show the usage and cost breakdown for the guardrail on the overview tab", async () => {
    renderDetail();
    const section = await screen.findByRole("region", { name: "Usage and cost" });
    expect(section).toHaveTextContent("$0.0004");
    expect(section).toHaveTextContent("Sensitive Information Policy");
  });

  it("should call onBack when 'Back to Overview' is clicked", async () => {
    const user = userEvent.setup();
    const onBack = vi.fn();
    renderDetail({ onBack });
    await user.click(await screen.findByRole("button", { name: /back to overview/i }));
    expect(onBack).toHaveBeenCalledOnce();
  });

  it("should offer an Overview tab and a Logs tab, with Overview selected first", async () => {
    renderDetail();
    expect(await screen.findByRole("tab", { name: "Overview" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("tab", { name: "Logs" })).toHaveAttribute("aria-selected", "false");
  });

  it("should select the Logs tab when it is clicked", async () => {
    const user = userEvent.setup();
    renderDetail();
    await user.click(await screen.findByRole("tab", { name: "Logs" }));
    await waitFor(() => expect(screen.getByRole("tab", { name: "Logs" })).toHaveAttribute("aria-selected", "true"));
    expect(screen.getByTestId("log-viewer")).toHaveTextContent("pii-detector");
  });

  it("should request the logs again when the date range changes", async () => {
    const { rerender } = renderDetail();
    await waitFor(() => expect(mockGetGuardrailsUsageLogs).toHaveBeenCalledTimes(1));

    rerender(<GuardrailDetail {...defaultProps} startDate="2026-06-01" endDate="2026-06-15" />);

    await waitFor(() => expect(mockGetGuardrailsUsageLogs).toHaveBeenCalledTimes(2));
    expect(mockGetGuardrailsUsageLogs).toHaveBeenLastCalledWith(
      "test-token",
      expect.objectContaining({ guardrailId: "pii-detector", startDate: "2026-06-01", endDate: "2026-06-15" }),
    );
  });

  it("should keep the evaluation settings modal closed until its button is clicked", async () => {
    const user = userEvent.setup();
    renderDetail();
    await screen.findByRole("heading", { name: "pii-detector" });
    expect(screen.queryByTestId("evaluation-modal")).not.toBeInTheDocument();

    await user.click(screen.getByTitle("Evaluation settings"));
    expect(screen.getByTestId("evaluation-modal")).toBeInTheDocument();
  });

  it("should not request anything without an access token", () => {
    mockUseGuardrailsUsageDetail.mockReturnValue(loaded(undefined));
    renderDetail({ accessToken: null });
    expect(mockUseGuardrailsUsageDetail).toHaveBeenCalledWith(
      "pii-detector",
      expect.objectContaining({ accessToken: null }),
    );
    expect(mockGetGuardrailsUsageLogs).not.toHaveBeenCalled();
  });

  describe("URL state", () => {
    it("should select the Logs tab when ?tab=logs", async () => {
      renderDetail({}, { searchParams: "?tab=logs" });

      expect(await screen.findByRole("tab", { name: "Logs" })).toHaveAttribute("aria-selected", "true");
      expect(screen.getByRole("tab", { name: "Overview" })).toHaveAttribute("aria-selected", "false");
    });

    it("should write the clicked tab to ?tab= and drop it for the overview", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<(event: UrlUpdateEvent) => void>();
      renderDetail({}, { onUrlUpdate });

      await user.click(await screen.findByRole("tab", { name: "Logs" }));
      await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("tab")).toBe("logs"));

      await user.click(screen.getByRole("tab", { name: "Overview" }));
      await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("tab")).toBe(false));
      expect(screen.getByRole("tab", { name: "Overview" })).toHaveAttribute("aria-selected", "true");
    });

    it("should hand the log filter, sample size and open request from the URL to the log viewer", async () => {
      renderDetail({}, { searchParams: "?tab=logs&log_filter=blocked&log_sample=50&request_id=req-3" });

      const viewer = await screen.findByTestId("log-viewer");
      expect(viewer).toHaveAttribute("data-filter", "blocked");
      expect(viewer).toHaveAttribute("data-sample", "50");
      expect(viewer).toHaveAttribute("data-request", "req-3");
    });

    it("should fall back to all logs and a sample of 10 for unknown values", async () => {
      renderDetail({}, { searchParams: "?log_filter=everything&log_sample=7" });

      const viewer = await screen.findByTestId("log-viewer");
      expect(viewer).toHaveAttribute("data-filter", "all");
      expect(viewer).toHaveAttribute("data-sample", "10");
      expect(viewer).toHaveAttribute("data-request", "");
    });

    it("should write the log viewer's filter, sample size and open request to the URL", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<(event: UrlUpdateEvent) => void>();
      renderDetail({}, { onUrlUpdate });

      await user.click(await screen.findByRole("button", { name: "show blocked" }));
      await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("log_filter")).toBe("blocked"));
      expect(screen.getByTestId("log-viewer")).toHaveAttribute("data-filter", "blocked");

      await user.click(screen.getByRole("button", { name: "sample 50" }));
      await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("log_sample")).toBe("50"));

      await user.click(screen.getByRole("button", { name: "open req-9" }));
      await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("request_id")).toBe("req-9"));
      expect(lastUrlUpdate(onUrlUpdate)?.options.history).toBe("push");

      await user.click(screen.getByRole("button", { name: "close request" }));
      await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("request_id")).toBe(false));
      expect(lastUrlUpdate(onUrlUpdate)?.options.history).toBe("replace");
      expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("log_filter")).toBe("blocked");
    });

    it("should drop the detail tab and log viewer state when going back", async () => {
      const user = userEvent.setup();
      const onBack = vi.fn();
      const onUrlUpdate = vi.fn<(event: UrlUpdateEvent) => void>();
      renderDetail(
        { onBack },
        {
          searchParams:
            "?guardrail=pii-detector&start_date=2026-07-01&tab=logs&log_filter=blocked&log_sample=100&request_id=req-3",
          onUrlUpdate,
        },
      );

      await user.click(await screen.findByRole("button", { name: /back to overview/i }));

      expect(onBack).toHaveBeenCalledOnce();
      await waitFor(() => expect(onUrlUpdate).toHaveBeenCalled());
      const update = lastUrlUpdate(onUrlUpdate);
      for (const key of ["tab", "log_filter", "log_sample", "request_id"]) {
        expect(update?.searchParams.has(key)).toBe(false);
      }
      expect(update?.searchParams.get("guardrail")).toBe("pii-detector");
      expect(update?.searchParams.get("start_date")).toBe("2026-07-01");
      expect(update?.options.history).toBe("replace");
    });
  });
});
