import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, screen, waitFor } from "@testing-library/react";
import { renderWithProviders } from "@/../tests/test-utils";
import AlertingSettings from "./alerting_settings";

const { alertingSettingsCall, updateConfigFieldSetting } = vi.hoisted(() => ({
  alertingSettingsCall: vi.fn(),
  updateConfigFieldSetting: vi.fn(),
}));

vi.mock("../networking", () => ({ alertingSettingsCall, updateConfigFieldSetting }));

const setting = (field_name: string, field_type: string, field_value: unknown) => ({
  field_name,
  field_type,
  field_value,
  field_default_value: field_value,
  field_description: `${field_name} description`,
  stored_in_db: true,
  premium_field: false,
});

const renderSettings = async (slackAlerting: boolean) => {
  alertingSettingsCall.mockResolvedValue([
    setting("daily_report_frequency", "Integer", 43200),
    setting("budget_alert_ttl", "Integer", 86400),
    setting("slack_alerting", "Boolean", slackAlerting),
  ]);
  renderWithProviders(
    <table>
      <tbody>
        <AlertingSettings accessToken="sk-test" premiumUser={false} />
      </tbody>
    </table>,
  );
  return screen.findByDisplayValue("43200");
};

describe("AlertingSettings save", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    updateConfigFieldSetting.mockResolvedValue({});
  });

  it("sends every alerting argument with the edited value as a number, then the slack destination", async () => {
    const frequency = await renderSettings(true);

    fireEvent.change(frequency, { target: { value: "7200" } });
    fireEvent.click(screen.getByRole("button", { name: "Update Settings" }));

    await waitFor(() => expect(updateConfigFieldSetting).toHaveBeenCalledTimes(2));
    expect(updateConfigFieldSetting).toHaveBeenNthCalledWith(1, "sk-test", "alerting_args", {
      daily_report_frequency: 7200,
      budget_alert_ttl: 86400,
    });
    expect(updateConfigFieldSetting).toHaveBeenNthCalledWith(2, "sk-test", "alerting", ["slack"]);
  });

  it("clears the alerting destinations when slack alerting is off", async () => {
    const frequency = await renderSettings(false);

    fireEvent.change(frequency, { target: { value: "7200" } });
    fireEvent.click(screen.getByRole("button", { name: "Update Settings" }));

    await waitFor(() => expect(updateConfigFieldSetting).toHaveBeenCalledTimes(2));
    expect(updateConfigFieldSetting).toHaveBeenNthCalledWith(2, "sk-test", "alerting", []);
  });
});
