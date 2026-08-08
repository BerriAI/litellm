import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "@/../tests/test-utils";
import EmailSettings from "./email_settings";

const { serviceHealthCheck, setCallbacksCall } = vi.hoisted(() => ({
  serviceHealthCheck: vi.fn(),
  setCallbacksCall: vi.fn(),
}));

vi.mock("@/components/networking", () => ({ serviceHealthCheck, setCallbacksCall }));

vi.mock("./email_events", () => ({
  EmailEventSettings: () => <div>email event settings</div>,
}));

const alerts = [
  {
    name: "email",
    variables: {
      SMTP_HOST: "smtp.example.com",
      SMTP_PORT: "587",
      SMTP_TLS: "True",
      SMTP_USE_SSL: "False",
      SMTP_USERNAME: "smtp-user",
      SMTP_PASSWORD: "********",
      SMTP_SENDER_EMAIL: "alerts@example.com",
      TEST_EMAIL_ADDRESS: "admin@example.com",
      EMAIL_LOGO_URL: "https://example.com/logo.png",
    },
  },
  { name: "slack", variables: { SLACK_WEBHOOK_URL: "https://hooks.example.com" } },
];

const fieldNamed = (name: string) => document.querySelector<HTMLInputElement | HTMLSelectElement>(`[name="${name}"]`)!;

describe("EmailSettings", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setCallbacksCall.mockResolvedValue({});
    serviceHealthCheck.mockResolvedValue({});
  });

  it("renders the heading and the docs link", () => {
    renderWithProviders(<EmailSettings accessToken="sk-test" premiumUser alerts={alerts} />);

    expect(screen.getByText("Email Server Settings")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /LiteLLM Docs: email alerts/ })).toHaveAttribute(
      "href",
      "https://docs.litellm.ai/docs/proxy/email",
    );
  });

  it("renders one named input per email variable and none for other alert types", () => {
    renderWithProviders(<EmailSettings accessToken="sk-test" premiumUser alerts={alerts} />);

    expect(fieldNamed("SMTP_HOST")).toHaveValue("smtp.example.com");
    expect(fieldNamed("SMTP_PORT")).toHaveValue(587);
    expect(fieldNamed("SMTP_PORT")).toHaveAttribute("type", "number");
    expect(fieldNamed("SMTP_HOST")).toHaveAttribute("type", "text");
    expect(fieldNamed("SMTP_TLS")).toHaveValue("True");
    expect(fieldNamed("SMTP_TLS").tagName).toBe("SELECT");
    expect(fieldNamed("SMTP_USE_SSL")).toHaveValue("False");
    expect(fieldNamed("SMTP_USE_SSL").tagName).toBe("SELECT");
    expect(fieldNamed("SMTP_USERNAME")).toHaveAttribute("type", "text");
    expect(fieldNamed("SMTP_PASSWORD")).toHaveValue("********");
    expect(fieldNamed("SMTP_SENDER_EMAIL")).toHaveAttribute("type", "text");
    expect(fieldNamed("TEST_EMAIL_ADDRESS")).toHaveAttribute("type", "text");
    expect(fieldNamed("EMAIL_LOGO_URL")).toHaveAttribute("type", "text");
    expect(document.querySelector('input[name="SLACK_WEBHOOK_URL"]')).toBeNull();
  });

  it("labels each variable and shows its help text", () => {
    renderWithProviders(<EmailSettings accessToken="sk-test" premiumUser alerts={alerts} />);

    expect(screen.getByText("SMTP_HOST")).toBeInTheDocument();
    expect(screen.getByText(/Enter the SMTP host address/)).toBeInTheDocument();
    expect(screen.getByText(/Enter the SMTP port number/)).toBeInTheDocument();
    expect(screen.getByText(/Optional SMTP username/)).toBeInTheDocument();
    expect(screen.getByText(/Optional SMTP password/)).toBeInTheDocument();
  });

  it("only requires the SMTP host and sender email", () => {
    renderWithProviders(<EmailSettings accessToken="sk-test" premiumUser alerts={alerts} />);

    expect(fieldNamed("SMTP_HOST")).toBeRequired();
    expect(fieldNamed("SMTP_SENDER_EMAIL")).toBeRequired();
    expect(fieldNamed("SMTP_PORT")).not.toBeRequired();
    expect(fieldNamed("SMTP_TLS")).not.toBeRequired();
    expect(fieldNamed("SMTP_USE_SSL")).not.toBeRequired();
    expect(fieldNamed("SMTP_USERNAME")).not.toBeRequired();
    expect(fieldNamed("SMTP_PASSWORD")).not.toBeRequired();
    expect(fieldNamed("TEST_EMAIL_ADDRESS")).not.toBeRequired();
  });

  it("submits only the fields the admin actually edited", async () => {
    const user = userEvent.setup();
    renderWithProviders(<EmailSettings accessToken="sk-test" premiumUser alerts={alerts} />);

    await user.clear(fieldNamed("SMTP_HOST"));
    await user.type(fieldNamed("SMTP_HOST"), "smtp.changed.com");
    await user.click(screen.getByRole("button", { name: "Save Changes" }));

    await waitFor(() => {
      expect(setCallbacksCall).toHaveBeenCalledWith("sk-test", {
        general_settings: { alerting: ["email"] },
        environment_variables: { SMTP_HOST: "smtp.changed.com" },
      });
    });
  });

  it("does not resubmit an untouched masked value", async () => {
    const user = userEvent.setup();
    renderWithProviders(<EmailSettings accessToken="sk-test" premiumUser alerts={alerts} />);

    await user.click(screen.getByRole("button", { name: "Save Changes" }));

    await waitFor(() => {
      expect(setCallbacksCall).toHaveBeenCalledWith("sk-test", {
        general_settings: { alerting: ["email"] },
        environment_variables: {},
      });
    });
  });

  it("submits empty values when optional credentials are cleared", async () => {
    const user = userEvent.setup();
    renderWithProviders(<EmailSettings accessToken="sk-test" premiumUser alerts={alerts} />);

    await user.clear(fieldNamed("SMTP_USERNAME"));
    await user.clear(fieldNamed("SMTP_PASSWORD"));
    await user.click(screen.getByRole("button", { name: "Save Changes" }));

    await waitFor(() => {
      expect(setCallbacksCall).toHaveBeenCalledWith("sk-test", {
        general_settings: { alerting: ["email"] },
        environment_variables: { SMTP_USERNAME: "", SMTP_PASSWORD: "" },
      });
    });
  });

  it("disables the premium-only fields for non-premium users", () => {
    renderWithProviders(<EmailSettings accessToken="sk-test" premiumUser={false} alerts={alerts} />);

    expect(fieldNamed("EMAIL_LOGO_URL")).toBeDisabled();
    expect(fieldNamed("SMTP_HOST")).not.toBeDisabled();
  });

  it("leaves the premium-only fields editable for premium users", () => {
    renderWithProviders(<EmailSettings accessToken="sk-test" premiumUser alerts={alerts} />);

    expect(fieldNamed("EMAIL_LOGO_URL")).not.toBeDisabled();
  });

  it("triggers a live email health check", async () => {
    const user = userEvent.setup();
    renderWithProviders(<EmailSettings accessToken="sk-test" premiumUser alerts={alerts} />);

    await user.click(screen.getByRole("button", { name: "Test Email Alerts" }));

    await waitFor(() => {
      expect(serviceHealthCheck).toHaveBeenCalledWith("sk-test", "email");
    });
  });

  it("renders the email event settings section", () => {
    renderWithProviders(<EmailSettings accessToken="sk-test" premiumUser alerts={alerts} />);

    expect(screen.getByText("email event settings")).toBeInTheDocument();
  });
});
