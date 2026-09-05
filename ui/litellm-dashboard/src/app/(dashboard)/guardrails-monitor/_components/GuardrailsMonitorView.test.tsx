import { type UrlUpdateEvent } from "nuqs/adapters/testing";
import { beforeEach, describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import GuardrailsMonitorView from "./GuardrailsMonitorView";
import * as networking from "@/components/networking";
import { renderWithProviders, screen, testQueryClient, waitFor } from "@/../tests/test-utils";

vi.mock("@/components/networking", () => ({
  getGuardrailsUsageOverview: vi.fn(),
  getGuardrailsUsageDetail: vi.fn(),
  getGuardrailsUsageLogs: vi.fn(),
  formatDate: vi.fn((d: Date) => d.toISOString().slice(0, 10)),
}));

vi.mock("@/components/GuardrailsMonitor/LogViewer", () => ({
  LogViewer: ({ guardrailName }: { guardrailName: string }) => <div data-testid="log-viewer">{guardrailName}</div>,
}));

const mockGetGuardrailsUsageOverview = vi.mocked(networking.getGuardrailsUsageOverview);
const mockGetGuardrailsUsageDetail = vi.mocked(networking.getGuardrailsUsageDetail);
const mockGetGuardrailsUsageLogs = vi.mocked(networking.getGuardrailsUsageLogs);

const emptyOverview = { rows: [], chart: [], totalRequests: 0, totalBlocked: 0, passRate: 100 };

const piiRow = {
  id: "gr-pii",
  name: "PII Guard",
  type: "pii",
  provider: "LiteLLM",
  requestsEvaluated: 10,
  failRate: 10,
  status: "healthy" as const,
  trend: "stable" as const,
};

const piiDetail = {
  guardrail_name: "PII Guard",
  description: "",
  status: "healthy",
  provider: "LiteLLM",
  type: "pii",
  requestsEvaluated: 10,
  failRate: 10,
  avgScore: 0.5,
  avgLatency: 20,
};

describe("GuardrailsMonitorView", () => {
  beforeEach(() => {
    testQueryClient.clear();
    vi.clearAllMocks();
    mockGetGuardrailsUsageOverview.mockResolvedValue(emptyOverview);
    mockGetGuardrailsUsageDetail.mockResolvedValue(piiDetail);
    mockGetGuardrailsUsageLogs.mockResolvedValue({ logs: [], total: 0 });
  });

  it("should render overview and fetch guardrails usage when accessToken is provided", async () => {
    renderWithProviders(<GuardrailsMonitorView accessToken="test-token" />);

    expect(await screen.findByRole("heading", { name: /Guardrails Monitor/i })).toBeInTheDocument();
    await waitFor(() => {
      expect(mockGetGuardrailsUsageOverview).toHaveBeenCalled();
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
      expect(mockGetGuardrailsUsageDetail).toHaveBeenCalledWith(
        "test-token",
        "gr-pii",
        expect.any(String),
        expect.any(String),
      );
      expect(screen.queryByRole("heading", { name: /Guardrails Monitor/i })).not.toBeInTheDocument();
    });

    it("should push ?guardrail= as a new history entry when a guardrail is selected", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<(event: UrlUpdateEvent) => void>();
      mockGetGuardrailsUsageOverview.mockResolvedValue({ ...emptyOverview, rows: [piiRow] });
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
});
