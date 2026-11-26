import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, screen, waitFor } from "@testing-library/react";
import { renderWithProviders } from "@/../tests/test-utils";
import MSTeamsSettings from "./MSTeamsSettings";

const { getCallbacksCall, serviceHealthCheck, setCallbacksCall } = vi.hoisted(() => ({
  getCallbacksCall: vi.fn(),
  serviceHealthCheck: vi.fn(),
  setCallbacksCall: vi.fn(),
}));

vi.mock("./networking", () => ({ getCallbacksCall, serviceHealthCheck, setCallbacksCall }));

const alerts = [
  {
    name: "ms_teams",
    variables: { MS_TEAMS_WEBHOOK_URL: "https://teams.example.com/old", MS_TEAMS_CHANNEL: "alerts" },
  },
  { name: "slack", variables: { SLACK_WEBHOOK_URL: "https://hooks.example.com" } },
];

const renderSettings = () =>
  renderWithProviders(<MSTeamsSettings accessToken="sk-test" userID="admin" userRole="Admin" alerts={alerts} />);

describe("MSTeamsSettings save", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    getCallbacksCall.mockResolvedValue({ active_alerting_destinations: ["slack"] });
    setCallbacksCall.mockResolvedValue({});
  });

  it("adds ms_teams to the current destinations and sends only the edited webhook", async () => {
    renderSettings();

    fireEvent.change(screen.getByDisplayValue("https://teams.example.com/old"), {
      target: { value: "https://teams.example.com/new" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Save Changes" }));

    await waitFor(() => expect(setCallbacksCall).toHaveBeenCalledTimes(1));
    expect(setCallbacksCall).toHaveBeenCalledWith("sk-test", {
      general_settings: { alerting: ["slack", "ms_teams"] },
      environment_variables: { MS_TEAMS_WEBHOOK_URL: "https://teams.example.com/new" },
    });
  });

  it("does not duplicate ms_teams or resubmit untouched values when it is already a destination", async () => {
    getCallbacksCall.mockResolvedValue({ active_alerting_destinations: ["ms_teams", "email"] });
    renderSettings();

    fireEvent.click(screen.getByRole("button", { name: "Save Changes" }));

    await waitFor(() => expect(setCallbacksCall).toHaveBeenCalledTimes(1));
    expect(setCallbacksCall).toHaveBeenCalledWith("sk-test", {
      general_settings: { alerting: ["ms_teams", "email"] },
      environment_variables: {},
    });
  });
});
