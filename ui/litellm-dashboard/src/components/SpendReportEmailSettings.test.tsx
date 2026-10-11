import { renderWithProviders, screen, waitFor, fireEvent } from "../../tests/test-utils";
import { vi } from "vitest";
import { SpendReportEmailSettings } from "./SpendReportEmailSettings";
import * as networking from "./networking";

vi.mock("./networking", () => ({
  getSpendReportEmailSettings: vi.fn(),
  getSpendReportEmailStatus: vi.fn().mockResolvedValue({ configured: true, provider: "smtp" }),
  updateSpendReportEmailSettings: vi.fn(),
  sendSpendReportEmailTest: vi.fn(),
  teamListCall: vi.fn().mockResolvedValue([]),
}));

describe("SpendReportEmailSettings", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders card title and description", async () => {
    vi.mocked(networking.getSpendReportEmailSettings).mockResolvedValue({
      values: {
        enabled: true,
        frequency: "both",
        daily_send_time: "08:30",
        monthly_send_time: "10:00",
        recipient_emails: ["finance@example.com"],
        group_by: ["team", "team_key_model"],
      },
      field_schema: {},
    });

    renderWithProviders(<SpendReportEmailSettings accessToken="test-token" defaultExpanded />);

    expect(screen.getByText("Scheduled Spend Reports (Email / SMTP)")).toBeInTheDocument();

    await waitFor(() => {
      expect(
        screen.getByText(
          "Automatically send daily and/or monthly spend reports to configured email addresses. Includes cumulative breakdowns by team, API keys, and models.",
        ),
      ).toBeInTheDocument();
      expect(screen.getAllByDisplayValue("finance@example.com").length).toBeGreaterThan(0);
    });
  });

  it("renders disabled state badge when enabled is false", async () => {
    vi.mocked(networking.getSpendReportEmailSettings).mockResolvedValue({
      values: {
        enabled: false,
        frequency: "daily",
        daily_send_time: "09:00",
        monthly_send_time: "09:00",
        recipient_emails: [],
        group_by: ["team"],
      },
      field_schema: {},
    });

    renderWithProviders(<SpendReportEmailSettings accessToken="test-token" />);

    await waitFor(() => {
      const disabledElements = screen.getAllByText("Disabled");
      expect(disabledElements.length).toBeGreaterThan(0);
    });
  });

  it("renders collapsed by default and allows expanding and collapsing", async () => {
    vi.mocked(networking.getSpendReportEmailSettings).mockResolvedValue({
      values: {
        enabled: true,
        frequency: "daily",
        daily_send_time: "09:00",
        monthly_send_time: "09:00",
        recipient_emails: ["ops@example.com"],
        group_by: ["team"],
      },
      field_schema: {},
    });

    renderWithProviders(<SpendReportEmailSettings accessToken="test-token" />);

    await waitFor(() => {
      expect(screen.getByText("Expand")).toBeInTheDocument();
      expect(screen.queryByText("Save Spend Report Settings")).not.toBeInTheDocument();
    });

    const expandButton = screen.getByRole("button", { name: /Expand spend report settings/i });
    fireEvent.click(expandButton);

    await waitFor(() => {
      expect(screen.getByText("Collapse")).toBeInTheDocument();
      expect(screen.getByText("Save Spend Report Settings")).toBeInTheDocument();
    });

    const collapseButton = screen.getByRole("button", { name: /Collapse spend report settings/i });
    fireEvent.click(collapseButton);

    await waitFor(() => {
      expect(screen.getByText("Expand")).toBeInTheDocument();
      expect(screen.queryByText("Save Spend Report Settings")).not.toBeInTheDocument();
    });
  });

  it("auto-expands when clicking Add Alert from collapsed state", async () => {
    vi.mocked(networking.getSpendReportEmailSettings).mockResolvedValue({
      values: {
        enabled: false,
        frequency: "daily",
        daily_send_time: "09:00",
        monthly_send_time: "09:00",
        recipient_emails: [],
        group_by: ["team"],
        alerts: [],
      },
      field_schema: {},
    });

    renderWithProviders(<SpendReportEmailSettings accessToken="test-token" />);

    await waitFor(() => {
      expect(screen.getByText("Expand")).toBeInTheDocument();
    });

    const addAlertButton = screen.getByRole("button", { name: "Add Alert" });
    fireEvent.click(addAlertButton);

    await waitFor(() => {
      expect(screen.getByText("Collapse")).toBeInTheDocument();
      expect(screen.getByDisplayValue("Spend Report Alert 2")).toBeInTheDocument();
      expect(screen.getByText("Save Spend Report Settings")).toBeInTheDocument();
    });
  });

  it("renders configured multi-alert list", async () => {
    vi.mocked(networking.getSpendReportEmailSettings).mockResolvedValue({
      values: {
        enabled: true,
        frequency: "daily",
        daily_send_time: "09:00",
        monthly_send_time: "09:00",
        recipient_emails: [],
        group_by: ["team"],
        alerts: [
          {
            id: "alert-daily",
            name: "Daily Executive Summary",
            enabled: true,
            frequency: "daily",
            send_time: "08:00",
            recipient_emails: ["exec@example.com"],
            group_by: ["team", "team_model"],
            team_id: null,
          },
          {
            id: "alert-monthly",
            name: "Monthly Team Alpha",
            enabled: false,
            frequency: "monthly",
            send_time: "10:00",
            recipient_emails: ["alpha-lead@example.com"],
            group_by: ["team_key_model"],
            team_id: "team-123",
          },
        ],
      },
      field_schema: {},
    });

    renderWithProviders(
      <SpendReportEmailSettings
        accessToken="test-token"
        defaultExpanded
        teams={[{ team_id: "team-123", team_alias: "Team Alpha" } as any]}
      />,
    );

    await waitFor(() => {
      expect(screen.getByDisplayValue("Daily Executive Summary")).toBeInTheDocument();
      expect(screen.getByDisplayValue("Monthly Team Alpha")).toBeInTheDocument();
      expect(screen.getByDisplayValue("exec@example.com")).toBeInTheDocument();
      expect(screen.getByDisplayValue("alpha-lead@example.com")).toBeInTheDocument();
      expect(screen.getByText("Team: Team Alpha")).toBeInTheDocument();
      expect(screen.getByText("Daily (Previous Day)")).toBeInTheDocument();
      expect(screen.getByText("Monthly (1st of Month)")).toBeInTheDocument();
      expect(screen.getByText("Team Alpha")).toBeInTheDocument();
    });
  });

  it("allows adding a new alert and saving", async () => {
    vi.mocked(networking.getSpendReportEmailSettings).mockResolvedValue({
      values: {
        enabled: true,
        frequency: "daily",
        daily_send_time: "09:00",
        monthly_send_time: "09:00",
        recipient_emails: [],
        group_by: ["team"],
        alerts: [
          {
            id: "alert-1",
            name: "Initial Alert",
            enabled: true,
            frequency: "daily",
            send_time: "09:00",
            recipient_emails: ["test@example.com"],
            group_by: ["team"],
            team_id: null,
          },
        ],
      },
      field_schema: {},
    });
    vi.mocked(networking.updateSpendReportEmailSettings).mockResolvedValue({});

    renderWithProviders(<SpendReportEmailSettings accessToken="test-token" defaultExpanded />);

    await waitFor(() => {
      expect(screen.getByDisplayValue("Initial Alert")).toBeInTheDocument();
    });

    const addButtons = screen.getAllByRole("button", { name: /Add Alert|Add Another Alert/i });
    fireEvent.click(addButtons[0]);

    await waitFor(() => {
      expect(screen.getByDisplayValue("Spend Report Alert 2")).toBeInTheDocument();
    });

    const saveButton = screen.getByRole("button", { name: "Save Spend Report Settings" });
    fireEvent.click(saveButton);

    await waitFor(() => {
      expect(networking.updateSpendReportEmailSettings).toHaveBeenCalledWith(
        "test-token",
        expect.objectContaining({
          alerts: expect.arrayContaining([
            expect.objectContaining({ name: "Initial Alert" }),
            expect.objectContaining({ name: "Spend Report Alert 2" }),
          ]),
        }),
      );
    });
  });

  it("allows deleting an alert", async () => {
    vi.mocked(networking.getSpendReportEmailSettings).mockResolvedValue({
      values: {
        enabled: true,
        frequency: "daily",
        daily_send_time: "09:00",
        monthly_send_time: "09:00",
        recipient_emails: [],
        group_by: ["team"],
        alerts: [
          {
            id: "alert-delete-me",
            name: "Alert To Delete",
            enabled: true,
            frequency: "daily",
            send_time: "09:00",
            recipient_emails: ["delete@example.com"],
            group_by: ["team"],
            team_id: null,
          },
        ],
      },
      field_schema: {},
    });

    renderWithProviders(<SpendReportEmailSettings accessToken="test-token" defaultExpanded />);

    await waitFor(() => {
      expect(screen.getByDisplayValue("Alert To Delete")).toBeInTheDocument();
    });

    const deleteButton = screen.getByRole("button", { name: "Delete alert Alert To Delete" });
    fireEvent.click(deleteButton);

    await waitFor(() => {
      expect(screen.queryByDisplayValue("Alert To Delete")).not.toBeInTheDocument();
      expect(screen.getByText("No scheduled spend report alerts configured")).toBeInTheDocument();
    });
  });

  it("sends a test email report for a specific alert", async () => {
    vi.mocked(networking.getSpendReportEmailSettings).mockResolvedValue({
      values: {
        enabled: true,
        frequency: "daily",
        daily_send_time: "09:00",
        monthly_send_time: "09:00",
        recipient_emails: [],
        group_by: ["team"],
        alerts: [
          {
            id: "alert-test",
            name: "Testable Alert",
            enabled: true,
            frequency: "monthly",
            send_time: "09:00",
            recipient_emails: ["ops@example.com"],
            group_by: ["team", "team_key"],
            team_id: "team-1",
          },
        ],
      },
      field_schema: {},
    });
    vi.mocked(networking.sendSpendReportEmailTest).mockResolvedValue({
      total_spend: 12.34,
      total_requests: 5,
    });

    renderWithProviders(<SpendReportEmailSettings accessToken="test-token" defaultExpanded />);

    await waitFor(() => {
      expect(screen.getByDisplayValue("Testable Alert")).toBeInTheDocument();
    });

    const testButton = screen.getByRole("button", { name: "Send Test Report Now" });
    fireEvent.click(testButton);

    await waitFor(() => {
      expect(networking.sendSpendReportEmailTest).toHaveBeenCalledWith(
        "test-token",
        expect.objectContaining({
          frequency: "monthly",
          recipient_emails: ["ops@example.com"],
          group_by: ["team", "team_key"],
          team_id: "team-1",
        }),
      );
    });
  });

  it("disables send test report button when email delivery is not configured", async () => {
    vi.mocked(networking.getSpendReportEmailSettings).mockResolvedValue({
      values: {
        enabled: true,
        frequency: "daily",
        daily_send_time: "09:00",
        monthly_send_time: "09:00",
        recipient_emails: ["ops@example.com"],
        group_by: ["team"],
        alerts: [
          {
            id: "alert-unconf",
            name: "Unconfigured Alert",
            enabled: true,
            frequency: "daily",
            send_time: "09:00",
            recipient_emails: ["ops@example.com"],
            group_by: ["team"],
            team_id: null,
          },
        ],
      },
      field_schema: {},
    });
    vi.mocked(networking.getSpendReportEmailStatus).mockResolvedValue({
      configured: false,
      provider: null,
    });

    renderWithProviders(<SpendReportEmailSettings accessToken="test-token" defaultExpanded />);

    await waitFor(() => {
      expect(screen.getByDisplayValue("Unconfigured Alert")).toBeInTheDocument();
    });

    const testButton = screen.getByRole("button", { name: "Send Test Report Now" });
    expect(testButton).toBeDisabled();
  });

  it("displays server timezone in the Send Time label", async () => {
    vi.mocked(networking.getSpendReportEmailSettings).mockResolvedValue({
      values: {
        enabled: true,
        frequency: "daily",
        daily_send_time: "09:00",
        monthly_send_time: "09:00",
        recipient_emails: ["ops@example.com"],
        group_by: ["team"],
        alerts: [
          {
            id: "alert-tz",
            name: "Timezone Alert",
            enabled: true,
            frequency: "daily",
            send_time: "09:00",
            recipient_emails: ["ops@example.com"],
            group_by: ["team"],
            team_id: null,
          },
        ],
      },
      field_schema: {},
      server_timezone: "CEST",
    });
    vi.mocked(networking.getSpendReportEmailStatus).mockResolvedValue({
      configured: true,
      provider: "smtp",
      server_timezone: "CEST",
    });

    renderWithProviders(<SpendReportEmailSettings accessToken="test-token" defaultExpanded />);

    await waitFor(() => {
      expect(screen.getByText("Send Time (CEST)")).toBeInTheDocument();
    });
  });
});
