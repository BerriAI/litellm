import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { screen, waitFor } from "@testing-library/react";
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
    await user.type(numberInputs[1], "5");
    await user.clear(numberInputs[1]);
    await user.click(screen.getByRole("switch", { name: "slack_alerting" }));
    await user.click(screen.getByRole("button", { name: "Update Settings" }));

    await waitFor(() => {
      expect(updateConfigFieldSetting).toHaveBeenNthCalledWith(1, "sk-test", "alerting_args", {
        budget_alert_ttl: 60,
      });
    });
    expect(updateConfigFieldSetting).toHaveBeenNthCalledWith(2, "sk-test", "alerting", ["slack"]);
  });
});
