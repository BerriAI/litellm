import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/app/(dashboard)/hooks/teams/useTeams", () => ({
  useInfiniteTeams: () => ({
    data: {
      pages: [
        {
          teams: [
            { team_id: "team-alpha", team_alias: "Alpha" },
            { team_id: "team-beta", team_alias: "Beta" },
          ],
        },
      ],
    },
    fetchNextPage: vi.fn(),
    hasNextPage: false,
    isFetchingNextPage: false,
    isLoading: false,
  }),
}));

vi.mock("@/components/ModelSelect/ModelSelect", () => ({
  ModelSelect: ({ onChange }: { onChange: (values: string[]) => void }) => (
    <button type="button" onClick={() => onChange(["all-proxy-models"])}>
      set-models
    </button>
  ),
}));

import NotificationsManager from "@/components/molecules/notifications_manager";

import { DefaultUserSettingsForm } from "./DefaultUserSettingsForm";
import type { InternalUserSettings } from "./mapper";

const POSSIBLE_UI_ROLES = {
  internal_user: { ui_label: "Internal User", description: "create and view own keys" },
  internal_user_viewer: { ui_label: "Internal Viewer", description: "view own keys" },
  proxy_admin: { ui_label: "Admin", description: "all permissions" },
};

const SETTINGS: InternalUserSettings = {
  values: {
    user_role: "internal_user",
    max_budget: 100,
    budget_duration: "30d",
    models: ["gpt-5.2"],
    teams: [{ team_id: "team-alpha", max_budget_in_team: 25, user_role: "user" }],
  },
  field_schema: {},
};

const SAVED_BODY = {
  user_role: "internal_user",
  max_budget: 100,
  budget_duration: "30d",
  models: ["gpt-5.2"],
  teams: [{ team_id: "team-alpha", max_budget_in_team: 25, user_role: "user" }],
};

const renderForm = (overrides?: {
  fetchSettings?: ReturnType<typeof vi.fn>;
  updateSettings?: ReturnType<typeof vi.fn>;
}) => {
  const fetchSettings = overrides?.fetchSettings ?? vi.fn().mockResolvedValue(SETTINGS);
  const updateSettings = overrides?.updateSettings ?? vi.fn().mockResolvedValue(undefined);
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });

  render(
    <QueryClientProvider client={queryClient}>
      <DefaultUserSettingsForm
        possibleUIRoles={POSSIBLE_UI_ROLES}
        fetchSettings={fetchSettings}
        updateSettings={updateSettings}
      />
    </QueryClientProvider>,
  );

  return { fetchSettings, updateSettings };
};

const saveButton = async () => await screen.findByRole("button", { name: "Save Changes" });

describe("DefaultUserSettingsForm", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("disables Save until the loaded settings are edited", async () => {
    renderForm();

    expect(await saveButton()).toBeDisabled();
  });

  it("shows an error instead of the form when the settings cannot be loaded", async () => {
    renderForm({ fetchSettings: vi.fn().mockRejectedValue(new Error("nope")) });

    expect(await screen.findByRole("alert")).toHaveTextContent("Could not load the default user settings.");
    expect(screen.queryByRole("button", { name: "Save Changes" })).not.toBeInTheDocument();
  });

  it("sends every field on save, not only the edited one", async () => {
    const user = userEvent.setup();
    const { updateSettings } = renderForm();

    await user.clear(await screen.findByLabelText("Max Budget (USD)"));
    await user.type(screen.getByLabelText("Max Budget (USD)"), "250");
    await user.click(await saveButton());

    await waitFor(() => expect(updateSettings).toHaveBeenCalledTimes(1));
    expect(updateSettings).toHaveBeenCalledWith({ ...SAVED_BODY, max_budget: 250 });
  });

  it("clears an emptied budget with null", async () => {
    const user = userEvent.setup();
    const { updateSettings } = renderForm();

    await user.clear(await screen.findByLabelText("Max Budget (USD)"));
    await user.click(await saveButton());

    await waitFor(() => expect(updateSettings).toHaveBeenCalledTimes(1));
    expect(updateSettings).toHaveBeenCalledWith({ ...SAVED_BODY, max_budget: null });
  });

  it("sends the models selection through unchanged, sentinel values included", async () => {
    const user = userEvent.setup();
    const { updateSettings } = renderForm();

    await user.click(await screen.findByRole("button", { name: "set-models" }));
    await user.click(await saveButton());

    await waitFor(() => expect(updateSettings).toHaveBeenCalledTimes(1));
    expect(updateSettings).toHaveBeenCalledWith({ ...SAVED_BODY, models: ["all-proxy-models"] });
  });

  it("saves a team that was picked from the searchable list", async () => {
    const user = userEvent.setup();
    const { updateSettings } = renderForm();

    await user.click(await screen.findByRole("button", { name: "Add Team" }));
    await user.click(screen.getAllByLabelText("Team")[1]);
    await user.click(await screen.findByText("Beta"));
    await user.click(await saveButton());

    await waitFor(() => expect(updateSettings).toHaveBeenCalledTimes(1));
    expect(updateSettings).toHaveBeenCalledWith({
      ...SAVED_BODY,
      teams: [
        { team_id: "team-alpha", max_budget_in_team: 25, user_role: "user" },
        { team_id: "team-beta", max_budget_in_team: null, user_role: "user" },
      ],
    });
  });

  it("never turns a team id typed into the picker into a saved team", async () => {
    const user = userEvent.setup();
    const { updateSettings } = renderForm();

    await user.click(await screen.findByRole("button", { name: "Add Team" }));
    await user.type(screen.getAllByLabelText("Team")[1], "team-alhpa");
    await user.keyboard("{Escape}");
    await user.click(await saveButton());

    expect(await screen.findByText("Select a team")).toBeInTheDocument();
    expect(screen.getAllByLabelText("Team")[1]).toHaveValue("");
    expect(updateSettings).not.toHaveBeenCalled();
  });

  it("blocks saving the same default team twice", async () => {
    const user = userEvent.setup();
    const { updateSettings } = renderForm();

    await user.click(await screen.findByRole("button", { name: "Add Team" }));
    await user.click(screen.getAllByLabelText("Team")[1]);
    await user.click(await screen.findByText("Alpha"));
    await user.click(await saveButton());

    expect(await screen.findByText("This team is already listed")).toBeInTheDocument();
    expect(updateSettings).not.toHaveBeenCalled();
  });

  it("drops a removed team row from the saved settings", async () => {
    const user = userEvent.setup();
    const { updateSettings } = renderForm();

    await user.click(await screen.findByRole("button", { name: "Remove" }));
    await user.click(await saveButton());

    await waitFor(() => expect(updateSettings).toHaveBeenCalledTimes(1));
    expect(updateSettings).toHaveBeenCalledWith({ ...SAVED_BODY, teams: null });
  });

  it("clears the dirty state after a successful save", async () => {
    const user = userEvent.setup();
    const { updateSettings } = renderForm();

    await user.clear(await screen.findByLabelText("Max Budget (USD)"));
    await user.type(screen.getByLabelText("Max Budget (USD)"), "250");
    await user.click(await saveButton());

    await waitFor(() => expect(updateSettings).toHaveBeenCalledTimes(1));
    await waitFor(async () => expect(await saveButton()).toBeDisabled());
    expect(NotificationsManager.success).toHaveBeenCalledWith("Default user settings updated successfully");
  });

  it("keeps the edit and surfaces the backend error when the save fails", async () => {
    const user = userEvent.setup();
    const { updateSettings } = renderForm({
      updateSettings: vi.fn().mockRejectedValue(new Error("Team(s) not found: team-alhpa.")),
    });

    await user.clear(await screen.findByLabelText("Max Budget (USD)"));
    await user.type(screen.getByLabelText("Max Budget (USD)"), "250");
    await user.click(await saveButton());

    await waitFor(() => expect(updateSettings).toHaveBeenCalledTimes(1));
    await waitFor(() =>
      expect(NotificationsManager.fromBackend).toHaveBeenCalledWith("Team(s) not found: team-alhpa."),
    );
    expect(await saveButton()).toBeEnabled();
    expect(screen.getByLabelText("Max Budget (USD)")).toHaveValue(250);
  });

  it("restores the loaded values when Cancel is pressed", async () => {
    const user = userEvent.setup();
    renderForm();

    await user.clear(await screen.findByLabelText("Max Budget (USD)"));
    await user.type(screen.getByLabelText("Max Budget (USD)"), "250");
    await user.click(screen.getByRole("button", { name: "Cancel" }));

    expect(screen.getByLabelText("Max Budget (USD)")).toHaveValue(100);
    expect(await saveButton()).toBeDisabled();
  });
});
