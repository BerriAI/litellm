import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/components/ModelSelect/ModelSelect", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/components/ModelSelect/ModelSelect")>();
  return {
    MODEL_SENTINEL_OPTIONS: actual.MODEL_SENTINEL_OPTIONS,
    ModelSelect: ({ onChange }: { onChange: (values: string[]) => void }) => (
      <button type="button" onClick={() => onChange(["all-proxy-models"])}>
        set-models
      </button>
    ),
  };
});

import NotificationsManager from "@/components/molecules/notifications_manager";

import { DefaultTeamSettingsForm } from "./DefaultTeamSettingsForm";
import type { DefaultTeamSettings } from "./mapper";

const SETTINGS: DefaultTeamSettings = {
  values: {
    max_budget: 100,
    budget_duration: "30d",
    tpm_limit: 1000,
    rpm_limit: 50,
    models: ["gpt-5.2"],
    team_member_permissions: ["/key/generate"],
  },
  field_schema: {},
};

const SAVED_BODY = {
  max_budget: 100,
  budget_duration: "30d",
  tpm_limit: 1000,
  rpm_limit: 50,
  models: ["gpt-5.2"],
  team_member_permissions: ["/key/generate"],
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
      <DefaultTeamSettingsForm fetchSettings={fetchSettings} updateSettings={updateSettings} />
    </QueryClientProvider>,
  );

  return { fetchSettings, updateSettings };
};

const saveButton = async () => await screen.findByRole("button", { name: "Save Changes" });

const enterEditMode = async (user: ReturnType<typeof userEvent.setup>) => {
  await user.click(await screen.findByRole("button", { name: "Edit Settings" }));
};

describe("DefaultTeamSettingsForm", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("shows a read-only summary until Edit Settings is clicked", async () => {
    renderForm();

    expect(await screen.findByText("100")).toBeInTheDocument();
    expect(screen.getByText("monthly")).toBeInTheDocument();
    expect(screen.getByText("1000")).toBeInTheDocument();
    expect(screen.getByText("50")).toBeInTheDocument();
    expect(screen.getByText("gpt-5.2")).toBeInTheDocument();
    expect(screen.getByText("/key/generate")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Save Changes" })).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Max Budget (USD)")).not.toBeInTheDocument();
  });

  it("labels model sentinels in the read-only summary", async () => {
    renderForm({
      fetchSettings: vi
        .fn()
        .mockResolvedValue({ ...SETTINGS, values: { ...SETTINGS.values, models: ["all-proxy-models"] } }),
    });

    expect(await screen.findByText("All Proxy Models")).toBeInTheDocument();
  });

  it("disables Save until the loaded settings are edited", async () => {
    const user = userEvent.setup();
    renderForm();

    await enterEditMode(user);

    expect(await saveButton()).toBeDisabled();
  });

  it("shows an error instead of the form when the settings cannot be loaded", async () => {
    renderForm({ fetchSettings: vi.fn().mockRejectedValue(new Error("nope")) });

    expect(await screen.findByRole("alert")).toHaveTextContent("Could not load the default team settings.");
    expect(screen.queryByRole("button", { name: "Edit Settings" })).not.toBeInTheDocument();
  });

  it("sends every field on save, not only the edited one", async () => {
    const user = userEvent.setup();
    const { updateSettings } = renderForm();

    await enterEditMode(user);
    await user.clear(await screen.findByLabelText("Max Budget (USD)"));
    await user.type(screen.getByLabelText("Max Budget (USD)"), "250");
    await user.click(await saveButton());

    await waitFor(() => expect(updateSettings).toHaveBeenCalledTimes(1));
    expect(updateSettings).toHaveBeenCalledWith({ ...SAVED_BODY, max_budget: 250 });
  });

  it("clears an emptied budget with null", async () => {
    const user = userEvent.setup();
    const { updateSettings } = renderForm();

    await enterEditMode(user);
    await user.clear(await screen.findByLabelText("Max Budget (USD)"));
    await user.click(await saveButton());

    await waitFor(() => expect(updateSettings).toHaveBeenCalledTimes(1));
    expect(updateSettings).toHaveBeenCalledWith({ ...SAVED_BODY, max_budget: null });
  });

  it("rejects a fractional TPM limit instead of saving it", async () => {
    const user = userEvent.setup();
    const { updateSettings } = renderForm();

    await enterEditMode(user);
    fireEvent.change(await screen.findByLabelText("TPM Limit"), { target: { value: "12.5" } });
    await user.click(await saveButton());

    expect(await screen.findByText("Must be a non-negative whole number")).toBeInTheDocument();
    expect(updateSettings).not.toHaveBeenCalled();
  });

  it("sends the models selection through unchanged, sentinel values included", async () => {
    const user = userEvent.setup();
    const { updateSettings } = renderForm();

    await enterEditMode(user);
    await user.click(await screen.findByRole("button", { name: "set-models" }));
    await user.click(await saveButton());

    await waitFor(() => expect(updateSettings).toHaveBeenCalledTimes(1));
    expect(updateSettings).toHaveBeenCalledWith({ ...SAVED_BODY, models: ["all-proxy-models"] });
  });

  it("saves a newly granted permission", async () => {
    const user = userEvent.setup();
    const { updateSettings } = renderForm();

    await enterEditMode(user);
    await user.click(await screen.findByRole("checkbox", { name: "/key/delete" }));
    await user.click(await saveButton());

    await waitFor(() => expect(updateSettings).toHaveBeenCalledTimes(1));
    expect(updateSettings).toHaveBeenCalledWith({
      ...SAVED_BODY,
      team_member_permissions: ["/key/generate", "/key/delete"],
    });
  });

  it("clears the permissions with null when the last one is revoked", async () => {
    const user = userEvent.setup();
    const { updateSettings } = renderForm();

    await enterEditMode(user);
    await user.click(await screen.findByRole("checkbox", { name: "/key/generate" }));
    await user.click(await saveButton());

    await waitFor(() => expect(updateSettings).toHaveBeenCalledTimes(1));
    expect(updateSettings).toHaveBeenCalledWith({ ...SAVED_BODY, team_member_permissions: null });
  });

  it("returns to the read-only view showing the new values after a successful save", async () => {
    const user = userEvent.setup();
    const updated = { ...SETTINGS, values: { ...SETTINGS.values, max_budget: 250 } };
    const { updateSettings } = renderForm({
      fetchSettings: vi.fn().mockResolvedValueOnce(SETTINGS).mockResolvedValue(updated),
    });

    await enterEditMode(user);
    await user.clear(await screen.findByLabelText("Max Budget (USD)"));
    await user.type(screen.getByLabelText("Max Budget (USD)"), "250");
    await user.click(await saveButton());

    await waitFor(() => expect(updateSettings).toHaveBeenCalledTimes(1));
    expect(await screen.findByRole("button", { name: "Edit Settings" })).toBeInTheDocument();
    expect(await screen.findByText("250")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Save Changes" })).not.toBeInTheDocument();
    expect(NotificationsManager.success).toHaveBeenCalledWith("Default team settings updated successfully");
  });

  it("keeps the edit and surfaces the backend error when the save fails", async () => {
    const user = userEvent.setup();
    const { updateSettings } = renderForm({
      updateSettings: vi.fn().mockRejectedValue(new Error("Set `'STORE_MODEL_IN_DB='True'` in your env.")),
    });

    await enterEditMode(user);
    await user.clear(await screen.findByLabelText("Max Budget (USD)"));
    await user.type(screen.getByLabelText("Max Budget (USD)"), "250");
    await user.click(await saveButton());

    await waitFor(() => expect(updateSettings).toHaveBeenCalledTimes(1));
    await waitFor(() =>
      expect(NotificationsManager.fromBackend).toHaveBeenCalledWith("Set `'STORE_MODEL_IN_DB='True'` in your env."),
    );
    expect(await saveButton()).toBeEnabled();
    expect(screen.getByLabelText("Max Budget (USD)")).toHaveValue(250);
  });

  it("discards edits and returns to the read-only view when Cancel is pressed", async () => {
    const user = userEvent.setup();
    const { updateSettings } = renderForm();

    await enterEditMode(user);
    await user.clear(await screen.findByLabelText("Max Budget (USD)"));
    await user.type(screen.getByLabelText("Max Budget (USD)"), "250");
    await user.click(screen.getByRole("button", { name: "Cancel" }));

    expect(await screen.findByRole("button", { name: "Edit Settings" })).toBeInTheDocument();
    expect(screen.getByText("100")).toBeInTheDocument();
    expect(updateSettings).not.toHaveBeenCalled();

    await enterEditMode(user);
    expect(screen.getByLabelText("Max Budget (USD)")).toHaveValue(100);
    expect(await saveButton()).toBeDisabled();
  });
});
