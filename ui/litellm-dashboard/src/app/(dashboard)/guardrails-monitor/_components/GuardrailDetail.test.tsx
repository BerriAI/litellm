import { describe, it, expect, vi, beforeEach } from "vitest";
import userEvent from "@testing-library/user-event";
import { render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { GuardrailDetail } from "./GuardrailDetail";

const mockGetGuardrailsUsageDetail = vi.fn();
const mockGetGuardrailsUsageLogs = vi.fn();
vi.mock("@/components/networking", () => ({
  getGuardrailsUsageDetail: (...args: unknown[]) => mockGetGuardrailsUsageDetail(...args),
  getGuardrailsUsageLogs: (...args: unknown[]) => mockGetGuardrailsUsageLogs(...args),
}));

vi.mock("@/components/GuardrailsMonitor/LogViewer", () => ({
  LogViewer: ({ guardrailName }: { guardrailName: string }) => <div data-testid="log-viewer">{guardrailName}</div>,
}));

vi.mock("./EvaluationSettingsModal", () => ({
  EvaluationSettingsModal: ({ open }: { open: boolean }) => (open ? <div data-testid="evaluation-modal" /> : null),
}));

const detail = {
  guardrail_name: "pii-detector",
  description: "Blocks personally identifiable information",
  status: "warning",
  provider: "presidio",
  type: "pii",
  requestsEvaluated: 12345,
  failRate: 20,
  avgScore: 0.4,
  avgLatency: 180,
};

const defaultProps = {
  guardrailId: "pii-detector",
  onBack: vi.fn(),
  accessToken: "test-token",
  startDate: "2026-07-01",
  endDate: "2026-07-24",
};

function renderDetail(props: Partial<typeof defaultProps> = {}) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<GuardrailDetail {...defaultProps} {...props} />, {
    wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>,
  });
}

describe("GuardrailDetail", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockGetGuardrailsUsageDetail.mockResolvedValue(detail);
    mockGetGuardrailsUsageLogs.mockResolvedValue({ logs: [], total: 0 });
  });

  it("should show a busy indicator while the detail request is in flight", () => {
    mockGetGuardrailsUsageDetail.mockReturnValue(new Promise(() => {}));
    renderDetail();
    expect(document.querySelector('[aria-busy="true"]')).toBeInTheDocument();
    expect(screen.queryByText("pii-detector")).not.toBeInTheDocument();
  });

  it("should show an error message and a way back when the detail request fails", async () => {
    mockGetGuardrailsUsageDetail.mockRejectedValue(new Error("boom"));
    renderDetail();
    expect(await screen.findByText("Failed to load guardrail details.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /back to overview/i })).toBeInTheDocument();
  });

  it("should request the detail and the logs for the guardrail and date range", async () => {
    renderDetail();
    await waitFor(() =>
      expect(mockGetGuardrailsUsageDetail).toHaveBeenCalledWith(
        "test-token",
        "pii-detector",
        "2026-07-01",
        "2026-07-24",
      ),
    );
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
    mockGetGuardrailsUsageDetail.mockResolvedValue({ ...detail, avgLatency: null });
    renderDetail();
    expect(await screen.findByText("No data")).toBeInTheDocument();
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

  it("should keep the evaluation settings modal closed until its button is clicked", async () => {
    const user = userEvent.setup();
    renderDetail();
    await screen.findByRole("heading", { name: "pii-detector" });
    expect(screen.queryByTestId("evaluation-modal")).not.toBeInTheDocument();

    await user.click(screen.getByTitle("Evaluation settings"));
    expect(screen.getByTestId("evaluation-modal")).toBeInTheDocument();
  });

  it("should not request anything without an access token", () => {
    renderDetail({ accessToken: null });
    expect(mockGetGuardrailsUsageDetail).not.toHaveBeenCalled();
    expect(mockGetGuardrailsUsageLogs).not.toHaveBeenCalled();
  });
});
