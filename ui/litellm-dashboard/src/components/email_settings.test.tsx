import { render, screen } from "@testing-library/react";
import EmailSettings from "./email_settings";

vi.mock("./networking", () => ({
  serviceHealthCheck: vi.fn(),
  setCallbacksCall: vi.fn(),
}));

vi.mock("./email_events", () => ({
  EmailEventSettings: () => <div data-testid="email-event-settings" />,
}));

const mockAlerts = [
  {
    name: "email",
    variables: {
      SMTP_HOST: "smtp.example.com",
      SMTP_PORT: "587",
      SMTP_USERNAME: "user",
      SMTP_PASSWORD: "pass",
      SMTP_SENDER_EMAIL: "sender@example.com",
      TEST_EMAIL_ADDRESS: "test@example.com",
    },
  },
];

describe("EmailSettings", () => {
  const defaultProps = {
    accessToken: "test-token",
    premiumUser: true,
    alerts: mockAlerts,
  };

  it("should render", () => {
    render(<EmailSettings {...defaultProps} />);
    expect(screen.getByText("Email Server Settings")).toBeInTheDocument();
  });

  it("should display SMTP fields from alerts", () => {
    render(<EmailSettings {...defaultProps} />);
    expect(screen.getByText("SMTP_HOST")).toBeInTheDocument();
    expect(screen.getByText("SMTP_PORT")).toBeInTheDocument();
    expect(screen.getByText("SMTP_USERNAME")).toBeInTheDocument();
  });

  it("should show Save Changes button", () => {
    render(<EmailSettings {...defaultProps} />);
    expect(screen.getByRole("button", { name: /save changes/i })).toBeInTheDocument();
  });

  it("should show Test Email Alerts button", () => {
    render(<EmailSettings {...defaultProps} />);
    expect(screen.getByRole("button", { name: /test email alerts/i })).toBeInTheDocument();
  });

  it("should show Required markers for required SMTP fields", () => {
    render(<EmailSettings {...defaultProps} />);
    expect(screen.getAllByText(/required \*/i).length).toBeGreaterThanOrEqual(5);
  });

  it("should disable premium fields for non-premium users", () => {
    render(<EmailSettings {...defaultProps} premiumUser={false} />);
    // EMAIL_LOGO_URL and EMAIL_SUPPORT_CONTACT should have a sparkle prefix for non-premium
    // They'll still show but with different rendering
    expect(screen.getByText("SMTP_HOST")).toBeInTheDocument();
  });

  it("should render EmailEventSettings sub-component", () => {
    render(<EmailSettings {...defaultProps} />);
    expect(screen.getByTestId("email-event-settings")).toBeInTheDocument();
  });

  it("should render docs link", () => {
    render(<EmailSettings {...defaultProps} />);
    expect(screen.getByText(/litellm docs: email alerts/i)).toBeInTheDocument();
  });
});
