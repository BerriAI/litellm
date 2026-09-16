import moment from "moment";
import { type UrlUpdateEvent } from "nuqs/adapters/testing";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import GuardrailsMonitorView from "./GuardrailsMonitorView";
import * as networking from "@/components/networking";
import { fireEvent, renderWithProviders, screen, testQueryClient, waitFor } from "@/../tests/test-utils";

vi.mock("@/components/networking", () => ({
  getGuardrailsUsageLogs: vi.fn(),
  formatDate: vi.fn((d: Date) => d.toISOString().slice(0, 10)),
}));

const mockUseGuardrailsUsageOverview = vi.fn();
const mockUseGuardrailsUsageDetail = vi.fn();
vi.mock("@/app/(dashboard)/hooks/guardrails/useGuardrailsUsage", () => ({
  useGuardrailsUsageOverview: (...args: unknown[]) => mockUseGuardrailsUsageOverview(...args),
  useGuardrailsUsageDetail: (...args: unknown[]) => mockUseGuardrailsUsageDetail(...args),
}));

vi.mock("@/components/GuardrailsMonitor/LogViewer", () => ({
  LogViewer: ({ guardrailName }: { guardrailName: string }) => <div data-testid="log-viewer">{guardrailName}</div>,
}));

const mockGetGuardrailsUsageLogs = vi.mocked(networking.getGuardrailsUsageLogs);

const emptyOverview = {
  rows: [],
  chart: [],
  totalRequests: 0,
  totalBlocked: 0,
  passRate: 100,
  totalUsageUnits: {},
  totalCost: null,
  totalUntrackedUsageUnits: {},
};

const piiRow = {
  id: "gr-pii",
  name: "PII Guard",
  type: "pii",
  provider: "LiteLLM",
  requestsEvaluated: 10,
  failRate: 10,
  avgScore: null,
  avgLatency: null,
  status: "healthy" as const,
  trend: "stable" as const,
  usageUnits: {},
  cost: null,
  untrackedUsageUnits: {},
};

const piiDetail = {
  guardrail_id: "gr-pii",
  guardrail_name: "PII Guard",
  description: "",
  status: "healthy",
  provider: "LiteLLM",
  type: "pii",
  requestsEvaluated: 10,
  failRate: 10,
  avgScore: 0.5,
  avgLatency: 20,
  trend: "stable",
  time_series: [],
  usage_units: {},
  usage_units_daily: [],
  usage_units_by_team: {},
  usage_units_by_key: {},
  cost: null,
  cost_by_unit: {},
  cost_by_team: {},
  cost_by_key: {},
  untracked_usage_units: {},
  untracked_usage_units_by_team: {},
  untracked_usage_units_by_key: {},
};

describe("GuardrailsMonitorView", () => {
  beforeEach(() => {
    testQueryClient.clear();
    vi.clearAllMocks();
    mockUseGuardrailsUsageOverview.mockReturnValue({ data: emptyOverview, isLoading: false, error: null });
    mockUseGuardrailsUsageDetail.mockReturnValue({ data: piiDetail, isLoading: false, error: null });
    mockGetGuardrailsUsageLogs.mockResolvedValue({ logs: [], total: 0 });
  });

  it("should render overview and fetch guardrails usage when accessToken is provided", async () => {
    renderWithProviders(<GuardrailsMonitorView accessToken="test-token" />);

    expect(await screen.findByRole("heading", { name: /Guardrails Monitor/i })).toBeInTheDocument();
    await waitFor(() => {
      expect(mockUseGuardrailsUsageOverview).toHaveBeenCalledWith(
        expect.objectContaining({ accessToken: "test-token", startDate: expect.any(String) }),
      );
    });
  });

  it("should render without crashing when accessToken is null", async () => {
    renderWithProviders(<GuardrailsMonitorView accessToken={null} />);
    expect(await screen.findByRole("heading", { name: /Guardrails Monitor/i })).toBeInTheDocument();
  });

  describe("guardrail detail deep link (?guardrail=)", () => {
    it("should open the detail view directly from a ?guardrail= deep link", async () => {
      renderWithProviders(<GuardrailsMonitorView accessToken="test-token" />, { searchParams: "?guardrail=gr-pii" });

      expect(await screen.findByRole("heading", { name: "PII Guard" })).toBeInTheDocument();
      expect(mockUseGuardrailsUsageDetail).toHaveBeenCalledWith(
        "gr-pii",
        expect.objectContaining({ accessToken: "test-token", startDate: expect.any(String) }),
      );
      expect(screen.queryByRole("heading", { name: /Guardrails Monitor/i })).not.toBeInTheDocument();
    });

    it("should push ?guardrail= as a new history entry when a guardrail is selected", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<(event: UrlUpdateEvent) => void>();
      mockUseGuardrailsUsageOverview.mockReturnValue({
        data: { ...emptyOverview, rows: [piiRow] },
        isLoading: false,
        error: null,
      });
      renderWithProviders(<GuardrailsMonitorView accessToken="test-token" />, { onUrlUpdate });

      await user.click(await screen.findByRole("button", { name: "PII Guard" }));

      await waitFor(() => expect(onUrlUpdate).toHaveBeenCalled());
      const lastUpdate = onUrlUpdate.mock.calls.at(-1)![0];
      expect(lastUpdate.searchParams.get("guardrail")).toBe("gr-pii");
      expect(lastUpdate.options.history).toBe("push");
      expect(await screen.findByRole("heading", { name: "PII Guard" })).toBeInTheDocument();
    });

    it("should clear ?guardrail= by replacing history when going back to the overview", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<(event: UrlUpdateEvent) => void>();
      renderWithProviders(<GuardrailsMonitorView accessToken="test-token" />, {
        searchParams: "?guardrail=gr-pii",
        onUrlUpdate,
      });

      await user.click(await screen.findByRole("button", { name: /back to overview/i }));

      await waitFor(() => expect(onUrlUpdate).toHaveBeenCalled());
      const lastUpdate = onUrlUpdate.mock.calls.at(-1)![0];
      expect(lastUpdate.searchParams.has("guardrail")).toBe(false);
      expect(lastUpdate.options.history).toBe("replace");
      expect(await screen.findByRole("heading", { name: /Guardrails Monitor/i })).toBeInTheDocument();
    });
  });

  describe("date range in the URL (?start_date= and ?end_date=)", () => {
    const lastOverviewRange = () => {
      const [options] = mockUseGuardrailsUsageOverview.mock.calls.at(-1) as [{ startDate: string; endDate: string }];
      return { startDate: options.startDate, endDate: options.endDate };
    };
    afterEach(() => {
      vi.unstubAllGlobals();
    });

    it("should request usage for the range in the URL and show it in the picker", () => {
      renderWithProviders(<GuardrailsMonitorView accessToken="test-token" />, {
        searchParams: "?start_date=2026-01-05&end_date=2026-01-20",
      });

      expect(lastOverviewRange()).toEqual({ startDate: "2026-01-05", endDate: "2026-01-20" });
      expect(screen.getByRole("button", { name: "5 Jan, 00:00 - 20 Jan, 23:59" })).toBeInTheDocument();
    });

    it("should pass the URL range to the detail view as well", () => {
      renderWithProviders(<GuardrailsMonitorView accessToken="test-token" />, {
        searchParams: "?guardrail=gr-pii&start_date=2026-01-05&end_date=2026-01-20",
      });

      expect(mockUseGuardrailsUsageDetail).toHaveBeenLastCalledWith(
        "gr-pii",
        expect.objectContaining({ startDate: "2026-01-05", endDate: "2026-01-20" }),
      );
    });

    it("should default to a seven day range ending today when the URL has none", () => {
      renderWithProviders(<GuardrailsMonitorView accessToken="test-token" />);

      const { startDate, endDate } = lastOverviewRange();
      expect(endDate).toBe(moment().format("YYYY-MM-DD"));
      expect(moment(endDate).diff(moment(startDate), "days")).toBe(7);
    });

    it("should ignore a malformed date and keep the other one", () => {
      renderWithProviders(<GuardrailsMonitorView accessToken="test-token" />, {
        searchParams: "?start_date=2026-02-30&end_date=2026-01-20",
      });

      const { startDate, endDate } = lastOverviewRange();
      expect(endDate).toBe("2026-01-20");
      expect(startDate).toBe(moment().subtract(7, "days").format("YYYY-MM-DD"));
    });

    it("should write the applied range to the URL and request usage for it", async () => {
      vi.stubGlobal("requestIdleCallback", (callback: () => void) => setTimeout(callback, 0));
      const onUrlUpdate = vi.fn<(event: UrlUpdateEvent) => void>();
      renderWithProviders(<GuardrailsMonitorView accessToken="test-token" />, {
        searchParams: "?start_date=2026-01-05&end_date=2026-01-20",
        onUrlUpdate,
      });

      fireEvent.click(screen.getByRole("button", { name: "5 Jan, 00:00 - 20 Jan, 23:59" }));
      fireEvent.change(screen.getByDisplayValue("2026-01-05"), { target: { value: "2026-03-01" } });
      fireEvent.change(screen.getByDisplayValue("2026-01-20"), { target: { value: "2026-03-15" } });
      fireEvent.click(screen.getByRole("button", { name: "Apply" }));

      await waitFor(() => expect(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.get("start_date")).toBe("2026-03-01"));
      expect(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.get("end_date")).toBe("2026-03-15");
      expect(lastOverviewRange()).toEqual({ startDate: "2026-03-01", endDate: "2026-03-15" });
      expect(screen.getByRole("button", { name: "1 Mar, 00:00 - 15 Mar, 23:59" })).toBeInTheDocument();
    });
  });
});
