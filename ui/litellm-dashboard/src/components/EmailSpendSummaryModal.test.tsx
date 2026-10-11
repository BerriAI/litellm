import { renderWithProviders, screen, waitFor, fireEvent } from "../../tests/test-utils";
import { vi } from "vitest";
import { EmailSpendSummaryModal } from "./EmailSpendSummaryModal";
import * as networking from "./networking";

vi.mock("./networking", () => ({
  sendSpendReportEmailTest: vi.fn(),
  getSpendReportEmailStatus: vi.fn(),
  formatDate: (d: Date) => d.toISOString().split("T")[0],
}));

describe("EmailSpendSummaryModal", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(networking.getSpendReportEmailStatus).mockResolvedValue({
      configured: true,
      provider: "smtp",
      message: "SMTP is configured",
    });
  });

  it("does not render dialog content when closed", () => {
    renderWithProviders(
      <EmailSpendSummaryModal
        isOpen={false}
        onClose={vi.fn()}
        accessToken="test-token"
        startDate="2026-03-01"
        endDate="2026-03-31"
      />,
    );

    expect(screen.queryByText("Email Spend Summary")).not.toBeInTheDocument();
  });

  it("renders modal with date range, settings link, and inputs when open", () => {
    renderWithProviders(
      <EmailSpendSummaryModal
        isOpen={true}
        onClose={vi.fn()}
        accessToken="test-token"
        startDate="2026-03-01"
        endDate="2026-03-31"
        teams={[{ team_id: "team-eng", team_alias: "Engineering" } as any]}
      />,
    );

    expect(screen.getByText("Email Spend Summary")).toBeInTheDocument();
    expect(screen.getByText("2026-03-01 to 2026-03-31")).toBeInTheDocument();
    expect(
      screen.getByText(/Want to receive spend summaries on a schedule/i),
    ).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Configure Scheduled Email Alerts/i })).toHaveAttribute(
      "href",
      "/logging-and-alerts?tab=email-alerts",
    );
  });

  it("submits spend summary request with parsed recipients and team filter", async () => {
    vi.mocked(networking.sendSpendReportEmailTest).mockResolvedValue({
      total_spend: 45.67,
      total_requests: 120,
    });
    const handleClose = vi.fn();

    renderWithProviders(
      <EmailSpendSummaryModal
        isOpen={true}
        onClose={handleClose}
        accessToken="test-token"
        startDate="2026-03-01"
        endDate="2026-03-31"
        initialTeamId="team-eng"
        teams={[{ team_id: "team-eng", team_alias: "Engineering" } as any]}
      />,
    );

    expect(screen.getByText("Engineering")).toBeInTheDocument();

    const input = screen.getByPlaceholderText(/finance@example.com/i);
    fireEvent.change(input, { target: { value: "alice@example.com, bob@example.com" } });

    const sendButton = screen.getByRole("button", { name: "Send Email Summary" });
    fireEvent.click(sendButton);

    await waitFor(() => {
      expect(networking.sendSpendReportEmailTest).toHaveBeenCalledWith(
        "test-token",
        expect.objectContaining({
          start_date: "2026-03-01",
          end_date: "2026-03-31",
          recipient_emails: ["alice@example.com", "bob@example.com"],
          team_id: "team-eng",
        }),
      );
      expect(handleClose).toHaveBeenCalled();
    });
  });

  it("disables send button when email delivery is not configured", () => {
    renderWithProviders(
      <EmailSpendSummaryModal
        isOpen={true}
        onClose={vi.fn()}
        accessToken="test-token"
        startDate="2026-03-01"
        endDate="2026-03-31"
        emailStatus={{
          configured: false,
          provider: null,
          message: "Email delivery is not configured",
        }}
      />,
    );

    expect(screen.getByRole("button", { name: "Send Email Summary" })).toBeDisabled();
  });
});
