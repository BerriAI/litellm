import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "@/../tests/test-utils";
import AlertingSettings from "./alerting_settings";

const { alertingSettingsCall, updateConfigFieldSetting } = vi.hoisted(() => ({
  alertingSettingsCall: vi.fn(),
  updateConfigFieldSetting: vi.fn(),
}));

vi.mock("@/components/networking", () => ({ alertingSettingsCall, updateConfigFieldSetting }));

vi.mock("@/lib/toast", () => ({
  toast: { error: vi.fn(), success: vi.fn() },
}));

const settingsResponse = [
  {
    field_name: "slack_alerting",
    field_type: "Boolean",
    field_value: false,
    field_default_value: null,
    field_description: "Enable Slack alerting",
    stored_in_db: null,
    premium_field: false,
  },
  {
    field_name: "budget_alert_ttl",
    field_type: "Integer",
    field_value: 60,
    field_default_value: 30,
    field_description: "Configured threshold",
    stored_in_db: true,
    premium_field: false,
  },
  {
    field_name: "outage_alert_ttl",
    field_type: "Integer",
    field_value: null,
    field_default_value: 10,
    field_description: "Unset threshold",
    stored_in_db: null,
    premium_field: false,
  },
];

describe("AlertingSettings", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    alertingSettingsCall.mockResolvedValue(settingsResponse);
    updateConfigFieldSetting.mockResolvedValue({});
  });

  it("does not submit unset fields when updating another alerting setting", async () => {
    const user = userEvent.setup();
    renderWithProviders(<AlertingSettings accessToken="sk-test" premiumUser />);

    await screen.findByDisplayValue("60");
    const numberInputs = screen.getAllByRole("spinbutton");
    fireEvent.change(numberInputs[1], { target: { value: "5" } });
    fireEvent.change(numberInputs[1], { target: { value: "" } });
    await user.click(screen.getByRole("switch", { name: "slack_alerting" }));
    await user.click(screen.getByRole("button", { name: "Update Settings" }));

    await waitFor(() => {
      expect(updateConfigFieldSetting).toHaveBeenNthCalledWith(1, "sk-test", "alerting_args", {
        budget_alert_ttl: 60,
      });
    });
    expect(updateConfigFieldSetting).toHaveBeenNthCalledWith(2, "sk-test", "alerting", ["slack"]);
  });

  it("does not restore a stale typed value after reset", async () => {
    const user = userEvent.setup();
    renderWithProviders(<AlertingSettings accessToken="sk-test" premiumUser />);

    const budgetInput = await screen.findByDisplayValue("60");
    fireEvent.change(budgetInput, { target: { value: "75" } });

    await user.click(screen.getByRole("button", { name: "Reset budget_alert_ttl" }));
    await user.click(screen.getByRole("button", { name: "Update Settings" }));

    await waitFor(() => {
      expect(updateConfigFieldSetting).toHaveBeenCalledWith("sk-test", "alerting_args", {});
    });
  });

  it("clears reset and form state when the access token changes", async () => {
    const refreshedSettings = settingsResponse.map((setting) =>
      setting.field_name === "budget_alert_ttl" ? { ...setting, field_value: 90 } : setting,
    );
    alertingSettingsCall.mockImplementation((token: string) =>
      Promise.resolve(token === "sk-new" ? refreshedSettings : settingsResponse),
    );

    const user = userEvent.setup();
    const { rerender } = renderWithProviders(<AlertingSettings accessToken="sk-old" premiumUser />);

    const budgetInput = await screen.findByDisplayValue("60");
    fireEvent.change(budgetInput, { target: { value: "75" } });
    await user.click(screen.getByRole("button", { name: "Reset budget_alert_ttl" }));

    rerender(<AlertingSettings accessToken="sk-new" premiumUser />);
    await screen.findByDisplayValue("90");

    await user.click(screen.getByRole("switch", { name: "slack_alerting" }));
    await user.click(screen.getByRole("button", { name: "Update Settings" }));

    await waitFor(() => {
      expect(updateConfigFieldSetting).toHaveBeenNthCalledWith(1, "sk-new", "alerting_args", {
        budget_alert_ttl: 90,
      });
    });
    expect(updateConfigFieldSetting).toHaveBeenNthCalledWith(2, "sk-new", "alerting", ["slack"]);
  });

  it("ignores an older settings response after the access token changes", async () => {
    const oldRequest = Promise.withResolvers<typeof settingsResponse>();
    const newRequest = Promise.withResolvers<typeof settingsResponse>();
    const refreshedSettings = settingsResponse.map((setting) =>
      setting.field_name === "budget_alert_ttl" ? { ...setting, field_value: 90 } : setting,
    );

    alertingSettingsCall.mockImplementation((token: string) =>
      token === "sk-new" ? newRequest.promise : oldRequest.promise,
    );

    const { rerender } = renderWithProviders(<AlertingSettings accessToken="sk-old" premiumUser />);
    rerender(<AlertingSettings accessToken="sk-new" premiumUser />);

    await act(async () => {
      newRequest.resolve(refreshedSettings);
      await newRequest.promise;
    });
    await screen.findByDisplayValue("90");

    await act(async () => {
      oldRequest.resolve(settingsResponse);
      await oldRequest.promise;
    });

    expect(screen.getByDisplayValue("90")).toBeInTheDocument();
    expect(screen.queryByDisplayValue("60")).not.toBeInTheDocument();
  });
});
