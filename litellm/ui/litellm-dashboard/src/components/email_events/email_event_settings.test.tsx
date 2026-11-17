import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "@/../tests/test-utils";
import EmailEventSettings from "./email_event_settings";

const { getEmailEventSettings, updateEmailEventSettings, resetEmailEventSettings } = vi.hoisted(() => ({
  getEmailEventSettings: vi.fn(),
  updateEmailEventSettings: vi.fn(),
  resetEmailEventSettings: vi.fn(),
}));

vi.mock("@/components/networking", () => ({
  getEmailEventSettings,
  updateEmailEventSettings,
  resetEmailEventSettings,
}));

const settingsResponse = {
  settings: [
    { event: "Virtual Key Created", enabled: true },
    { event: "New User Invitation", enabled: false },
  ],
};

describe("EmailEventSettings", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    getEmailEventSettings.mockResolvedValue(settingsResponse);
    updateEmailEventSettings.mockResolvedValue({});
    resetEmailEventSettings.mockResolvedValue({});
  });

  it("renders the heading and the explanatory copy", async () => {
    renderWithProviders(<EmailEventSettings accessToken="sk-test" />);

    expect(await screen.findByText("Email Notifications")).toBeInTheDocument();
    expect(screen.getByText("Select which events should trigger email notifications.")).toBeInTheDocument();
  });

  it("should render one toggle per event, reflecting the persisted enabled state", async () => {
    renderWithProviders(<EmailEventSettings accessToken="sk-test" />);

    await screen.findByText("Virtual Key Created");
    const toggles = screen.getAllByRole("switch");

    expect(toggles).toHaveLength(2);
    expect(toggles[0]).toBeChecked();
    expect(toggles[1]).not.toBeChecked();
  });

  it("renders a per-event description", async () => {
    renderWithProviders(<EmailEventSettings accessToken="sk-test" />);

    expect(
      await screen.findByText(/An email will be sent to the user when a new virtual key is created/),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/An email will be sent to the email address of the user when a new user is created/),
    ).toBeInTheDocument();
  });

  it("saves the toggled enabled flags rather than the originally fetched ones", async () => {
    const user = userEvent.setup();
    renderWithProviders(<EmailEventSettings accessToken="sk-test" />);

    await screen.findByText("Virtual Key Created");
    await user.click(screen.getAllByRole("switch")[1]);
    await user.click(screen.getByRole("button", { name: "Save Changes" }));

    await waitFor(() => {
      expect(updateEmailEventSettings).toHaveBeenCalledWith("sk-test", {
        settings: [
          { event: "Virtual Key Created", enabled: true },
          { event: "New User Invitation", enabled: true },
        ],
      });
    });
  });

  it("unchecking an enabled event is persisted as disabled", async () => {
    const user = userEvent.setup();
    renderWithProviders(<EmailEventSettings accessToken="sk-test" />);

    await screen.findByText("Virtual Key Created");
    await user.click(screen.getAllByRole("switch")[0]);
    await user.click(screen.getByRole("button", { name: "Save Changes" }));

    await waitFor(() => {
      expect(updateEmailEventSettings).toHaveBeenCalledWith("sk-test", {
        settings: [
          { event: "Virtual Key Created", enabled: false },
          { event: "New User Invitation", enabled: false },
        ],
      });
    });
  });

  it("resets to defaults and refetches the settings", async () => {
    const user = userEvent.setup();
    renderWithProviders(<EmailEventSettings accessToken="sk-test" />);

    await screen.findByText("Virtual Key Created");
    expect(getEmailEventSettings).toHaveBeenCalledTimes(1);

    await user.click(screen.getByRole("button", { name: "Reset to Defaults" }));

    await waitFor(() => {
      expect(resetEmailEventSettings).toHaveBeenCalledWith("sk-test");
    });
    await waitFor(() => {
      expect(getEmailEventSettings).toHaveBeenCalledTimes(2);
    });
  });

  it("does not render event rows while the fetch is in flight", () => {
    getEmailEventSettings.mockReturnValue(new Promise(() => {}));
    renderWithProviders(<EmailEventSettings accessToken="sk-test" />);

    expect(screen.queryByRole("switch")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Save Changes" })).toBeDisabled();
  });

  it("does not call the API when there is no access token", async () => {
    renderWithProviders(<EmailEventSettings accessToken={null} />);

    await waitFor(() => {
      expect(getEmailEventSettings).not.toHaveBeenCalled();
    });
  });
  it("should show model deprecation reminders off and save explicit opt-in", async () => {
    const user = userEvent.setup();
    getEmailEventSettings.mockResolvedValue({ settings: [{ event: "Model Deprecations", enabled: false }] });
    renderWithProviders(<EmailEventSettings accessToken="test-access" />);
    const toggle = await screen.findByRole("switch", { name: "Model Deprecations" });
    expect(toggle).not.toBeChecked();
    expect(screen.queryByRole("checkbox")).not.toBeInTheDocument();
    expect(screen.getByText(/30 days and 1 day before model deprecation/)).toBeInTheDocument();
    await user.click(toggle);
    await user.click(screen.getByRole("button", { name: "Save Changes" }));
    await waitFor(() =>
      expect(updateEmailEventSettings).toHaveBeenCalledWith("test-access", {
        settings: [{ event: "Model Deprecations", enabled: true }],
      }),
    );
  });
});
